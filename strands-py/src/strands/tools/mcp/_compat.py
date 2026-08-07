"""Compatibility layer over the `mcp` 1.x and 2.x lines.

The official `mcp` package renamed and relocated several public names in 2.0.
Import any version-dependent name from this module instead of from `mcp`
directly so the rest of the codebase stays version-agnostic. Branch on
`MCP_V2` only where behavior differs between the two lines; pure renames are
resolved here once.
"""

from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
from mcp import ClientSession

__all__ = ["MCP_V2", "GetSessionIdCallback", "MCPError", "streamable_http_transport"]

# Feature-probed rather than version-parsed so pre-releases and backports
# resolve by capability: `ClientSession.discover` is the 2.x replacement for
# the removed initialize handshake.
MCP_V2: bool = hasattr(ClientSession, "discover")

try:
    from mcp.shared.exceptions import MCPError
except ImportError:
    # mcp 1.x spells the class McpError
    from mcp.shared.exceptions import McpError as MCPError  # type: ignore[attr-defined, no-redef]

try:
    from mcp.client.streamable_http import GetSessionIdCallback
except ImportError:
    # mcp 2.x removed protocol sessions, so its transports never yield a
    # session-id callback; the alias survives only to type 1.x transports.
    from collections.abc import Callable

    GetSessionIdCallback = Callable[[], str | None]  # type: ignore[misc, assignment]


def streamable_http_transport(
    url: str, headers: dict[str, Any] | None = None, auth: httpx.Auth | None = None
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
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        return streamable_http_client(url=url, http_client=create_mcp_http_client(headers=headers, auth=auth))

    from mcp.client.streamable_http import streamablehttp_client

    return streamablehttp_client(url=url, headers=headers, auth=auth)
