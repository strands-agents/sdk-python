"""Agent-callable MCP client tool.

Provides :func:`make_mcp_client` (a factory bound to a developer-set server allowlist).

The tool exposes four commands — ``connect``, ``list_tools``, ``call_tool``,
``disconnect`` — letting an agent open a connection to an MCP server, discover its
tools, invoke one, and close the connection. Each server is configured with a
:class:`~strands.tools.mcp.MCPServerConfig`; all fields are forwarded to
:class:`~strands.tools.mcp.MCPClient`. One connection is held per agent at a time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

from ...tools.decorator import tool
from ...tools.mcp.mcp_client import MCPClient, MCPServerConfig
from ...tools.mcp.mcp_types import MCPToolResult
from ...types.tools import ToolContext, ToolSpec
from .types import MCP_CLIENT_DESCRIPTION, Command

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)

_DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}
_ALLOWED_SCHEMES = frozenset(_DEFAULT_SCHEME_PORTS)


class MCPClientToolError(RuntimeError):
    """Raised when an mcp_client tool operation fails."""


def make_mcp_client(
    *,
    name: str = "mcp_client",
    description: str | None = None,
    servers: list[MCPServerConfig],
) -> DecoratedFunctionTool:
    """Create an agent-callable MCP client tool bound to a developer-set server allowlist.

    Args:
        name: Tool name. Defaults to ``"mcp_client"``.
        description: Tool description shown to the model. Defaults to a description
            that includes the list of permitted servers.
        servers: Server configurations the tool may connect to. Each entry must have either
            a ``url`` field (streamable-http) or a ``command`` field (stdio). Must not be empty.

    Returns:
        A decorated tool that manages MCP sessions.

    Raises:
        ValueError: If ``servers`` is empty or contains an invalid entry.
    """
    server_map = _validate_servers(servers)

    if description is None:
        permitted = ", ".join(f"'{s}'" for s in sorted(server_map))
        description = f"{MCP_CLIENT_DESCRIPTION} Permitted servers: {permitted}."

    # One MCPClient per agent. WeakKeyDictionary so agents can be garbage collected normally.
    clients: weakref.WeakKeyDictionary[Any, MCPClient] = weakref.WeakKeyDictionary()

    @tool(name=name, description=description, context="tool_context")
    async def mcp_client_tool(
        command: Command,
        tool_context: ToolContext,
        server: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> list[ToolSpec] | MCPToolResult | str:
        """Manage a runtime MCP client connection.

        Args:
            command: The operation to perform: ``connect``, ``list_tools``, ``call_tool``, ``disconnect``.
            tool_context: Injected by the framework. Not user-facing.
            server: Server to connect to for ``connect``. For HTTP servers, a URL; for stdio servers,
                the full command invocation. Must match the developer-set allowlist verbatim.
            tool_name: Tool name to invoke, required for ``call_tool``.
            arguments: Arguments to pass to the invoked tool, for ``call_tool``.

        Raises:
            MCPClientToolError: If a required argument is missing, the server is not on the allowlist,
                no connection is active, or the connection fails to start.
        """
        agent = tool_context.agent

        if command == "connect":
            if not server:
                raise MCPClientToolError("`server` is required for command='connect'")
            return await _handle_connect(clients, agent, server_map=server_map, server=server)

        client = clients.get(agent)
        if client is None:
            raise MCPClientToolError("No active connection. Call 'connect' first.")

        if command == "list_tools":
            return await _handle_list_tools(client)

        if command == "call_tool":
            if not tool_name:
                raise MCPClientToolError("`tool_name` is required for command='call_tool'")
            return await client.call_tool_async(
                tool_use_id=str(uuid4()),
                name=tool_name,
                arguments=arguments,
                cancel_signal=tool_context.cancel_signal,
            )

        if command == "disconnect":
            return await _handle_disconnect(clients, agent)

        raise MCPClientToolError(f"Unknown command: {command}")

    return mcp_client_tool


# ---- Internals ----------------------------------------------------------------


def _stop_client_on_gc(client: MCPClient) -> None:
    """Stop an MCPClient from a GC finalizer; runs stop() in a daemon thread with a 1 s cap."""
    thread = threading.Thread(target=client.stop, args=(None, None, None), daemon=True)
    thread.start()
    thread.join(timeout=1.0)


def _validate_servers(servers: list[MCPServerConfig]) -> dict[str, MCPServerConfig]:
    """Validate server configs and return a map keyed by server identifier.

    For HTTP entries the key is the canonicalised URL; for stdio entries it is
    the full command invocation (``command`` + space-joined ``args``).

    Returns:
        A dict keyed by server identifier mapping to the original config.

    Raises:
        ValueError: If ``servers`` is empty, or an HTTP entry has an invalid URL.
    """
    if not servers:
        raise ValueError("`servers` must not be empty; the mcp_client tool requires at least one server")

    server_map: dict[str, MCPServerConfig] = {}

    for config in servers:
        if config.get("disabled"):
            raise ValueError(
                f"Server config with url={config.get('url')!r} command={config.get('command')!r} "
                "is disabled; remove it from the list or set disabled=False"
            )
        url = config.get("url")
        command = config.get("command")
        if url:
            if command:
                raise ValueError(
                    f"Server config has both 'url' ({url!r}) and 'command' ({command!r}); "
                    "provide one or the other"
                )
            parsed = urlsplit(url)
            if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
                raise ValueError(
                    f"Server URL {url!r} has unsupported scheme {parsed.scheme!r}; only http and https are supported"
                )
            if not parsed.hostname:
                raise ValueError(f"Server URL {url!r} has no host")
            key = _canonicalise_url(url)
        elif command:
            key = " ".join([command] + list(config.get("args") or []))
        else:
            raise ValueError("Each server config must have either a 'url' (HTTP) or 'command' (stdio) field")
        if key in server_map and server_map[key] != config:
            raise ValueError(
                f"Server key {key!r} is produced by two different configs; "
                "remove the duplicate or disambiguate (e.g. use explicit ports or distinct commands)"
            )
        server_map[key] = config

    return server_map


def _canonicalise_url(url: str) -> str:
    """Canonicalise a URL for allowlist matching.

    Lowercase scheme/host, strip the trailing slash on the path, drop the port when it
    matches the scheme default (so ``https://host`` and ``https://host:443`` match), and
    drop userinfo and fragment.
    """
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    # Re-bracket IPv6 addresses stripped by urlsplit
    if ":" in host:
        host = f"[{host}]"
    default_port = _DEFAULT_SCHEME_PORTS.get(scheme)
    port = f":{parsed.port}" if parsed.port and parsed.port != default_port else ""
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{host}{port}{path}{query}"


async def _handle_connect(
    clients: weakref.WeakKeyDictionary[Any, MCPClient],
    agent: Any,
    *,
    server_map: dict[str, MCPServerConfig],
    server: str,
) -> str:
    is_http = server.lower().startswith(("http://", "https://"))
    try:
        key = _canonicalise_url(server) if is_http else server
    except ValueError:
        permitted = ", ".join(sorted(server_map))
        raise MCPClientToolError(f"Server {server!r} is not a valid URL. Permitted servers: {permitted}") from None

    if key not in server_map:
        permitted = ", ".join(sorted(server_map))
        raise MCPClientToolError(f"Server {server!r} is not on the allowlist. Permitted servers: {permitted}")

    # Stop any existing connection before opening a new one.
    existing = clients.get(agent)
    if existing is not None:
        await _handle_disconnect(clients, agent)

    config = cast(dict[str, Any], server_map[key])
    loaded = MCPClient.load_servers({"vended": config})
    client = loaded[0]

    try:
        await asyncio.to_thread(client.start)
        clients[agent] = client
        weakref.finalize(agent, _stop_client_on_gc, client)
    except BaseException:
        try:
            client.stop(None, None, None)
        except Exception:
            logger.debug("failed to stop MCP client after connect failure", exc_info=True)
        raise

    logger.debug("server=<%s> | opened MCP connection", key)
    return f"Successfully connected to {key}"


async def _handle_list_tools(client: MCPClient) -> list[ToolSpec]:
    """Return the tool list from the client, using server-side names so call_tool can invoke them directly."""
    agent_tools = await asyncio.to_thread(
        lambda: client._list_all_tools_sync()  # noqa: SLF001
    )
    return [{**t.tool_spec, "name": t.mcp_tool.name} for t in agent_tools]


async def _handle_disconnect(
    clients: weakref.WeakKeyDictionary[Any, MCPClient],
    agent: Any,
) -> str:
    client = clients.get(agent)
    if client is not None:
        try:
            await asyncio.to_thread(client.stop, None, None, None)
        except RuntimeError:
            logger.debug("MCP connection was already closed")
        finally:
            clients.pop(agent, None)
    return "Successfully disconnected"
