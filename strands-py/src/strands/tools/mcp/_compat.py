"""Compatibility layer over the `mcp` 1.x and 2.x lines.

The official `mcp` package renamed and relocated several public names in 2.0.
Import any version-dependent name from this module instead of from `mcp`
directly so the rest of the codebase stays version-agnostic. Branch on
`MCP_V2` only where behavior differs between the two lines; pure renames are
resolved here once.

mypy note: only one `mcp` line is installed at a time, so each try/except
branch below is an `attr-defined` error when checked against the other line.
The branch-level ignores cover whichever line mypy runs under, and the
per-module `warn_unused_ignores` override in pyproject.toml silences the
ignores the installed line doesn't need.
"""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.types import ServerCapabilities

__all__ = ["MCP_V2", "GetSessionIdCallback", "MCPError", "initialize_session", "streamable_http_transport"]

# Feature-probed rather than version-parsed so pre-releases and backports
# resolve by capability: `ClientSession.discover` is the 2.x replacement for
# the removed initialize handshake.
MCP_V2: bool = hasattr(ClientSession, "discover")

try:
    from mcp.shared.exceptions import MCPError  # type: ignore[attr-defined]
except ImportError:
    # mcp 1.x spells the class McpError
    from mcp.shared.exceptions import McpError as MCPError  # type: ignore[attr-defined, no-redef]

try:
    from mcp.client.streamable_http import GetSessionIdCallback  # type: ignore[attr-defined]
except ImportError:
    # mcp 2.x removed protocol sessions, so its transports never yield a
    # session-id callback; the alias survives only to type 1.x transports.
    from collections.abc import Callable

    GetSessionIdCallback = Callable[[], str | None]  # type: ignore[misc, assignment]


async def initialize_session(session: ClientSession) -> tuple[str | None, ServerCapabilities | None]:
    """Negotiate the connection on an entered session, on either `mcp` major line.

    The 2026-07-28 spec replaced the mandatory `initialize` handshake with a
    stateless `server/discover` probe (SEP-2575). `negotiate_auto` is the
    official 2.x client's connect-time policy: it probes `server/discover`
    first and falls back to the legacy handshake for pre-2026 servers, so
    either server era works. It lives in a private module, so the forced-2.x
    CI job is what catches a relocation.

    Args:
        session: An entered `ClientSession` that has not yet negotiated.

    Returns:
        The server's instructions (if any) and advertised capabilities.
    """
    if MCP_V2:
        from mcp.client._probe import negotiate_auto  # type: ignore[import-not-found]

        await negotiate_auto(session)
        return session.instructions, session.server_capabilities  # type: ignore[attr-defined]

    init_result = await session.initialize()
    return init_result.instructions, session.get_server_capabilities()


def streamable_http_transport(
    url: str, headers: dict[str, str] | None = None, auth: httpx.Auth | None = None
) -> AbstractAsyncContextManager[Any]:
    """Open a streamable HTTP client transport on either `mcp` major line.

    `mcp` 2.x replaced `streamablehttp_client(url, headers=..., auth=...)` with
    `streamable_http_client(url, http_client=...)`, which takes a
    pre-configured HTTPX client instead of loose header and auth kwargs. This
    adapter keeps the 1.x-style call shape for both lines.

    The imports are resolved at call time because each name exists on only
    one major line.

    Args:
        url: The MCP server endpoint URL.
        headers: Optional HTTP headers to send with each request.
        auth: Optional HTTPX authentication handler for each request.

    Returns:
        An async context manager yielding the transport's read/write streams.
    """
    if MCP_V2:
        from mcp.client.streamable_http import (  # type: ignore[attr-defined]
            create_mcp_http_client,
            streamable_http_client,
        )

        # `streamable_http_client` closes an HTTPX client only when it created
        # it (`client_provided` check in mcp 2.x), so a caller-provided client
        # must be closed by the caller: enter both context managers together
        # so the client's lifetime is bound to the transport's.
        @asynccontextmanager
        async def _owned_client_transport() -> AsyncIterator[Any]:
            async with (
                create_mcp_http_client(headers=headers, auth=auth) as http_client,
                streamable_http_client(url=url, http_client=http_client) as transport_streams,
            ):
                yield transport_streams

        return _owned_client_transport()

    from mcp.client.streamable_http import streamablehttp_client  # type: ignore[attr-defined]

    return streamablehttp_client(url=url, headers=headers, auth=auth)  # type: ignore[no-any-return]
