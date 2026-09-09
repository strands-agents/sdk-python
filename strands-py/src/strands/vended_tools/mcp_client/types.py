"""Shared types and constants for the mcp_client tool."""

from __future__ import annotations

from typing import Literal, TypedDict

from ...tools.mcp.mcp_client import MCPClient

Command = Literal["connect", "list_tools", "call_tool", "disconnect"]
"""Commands supported by the mcp_client tool."""


class ConnectOutput(TypedDict):
    """Output of the ``connect`` command.

    Attributes:
        session_id: Opaque identifier for the new session. Pass to subsequent commands.
        server: The server identifier used to open the session (canonical URL or full command invocation).
    """

    session_id: str
    server: str


class _Session:
    """A single live MCP connection."""

    __slots__ = ("client", "key")

    def __init__(self, client: MCPClient, key: str) -> None:
        self.client = client
        self.key = key


MCP_CLIENT_DESCRIPTION = (
    "Connects to Model Context Protocol (MCP) servers at runtime. Supports four commands: "
    "'connect' (open a session to a server — a URL for HTTP servers or a command invocation for stdio), "
    "'list_tools' (list the tools the connected server exposes), "
    "'call_tool' (invoke a tool on a connected server), and "
    "'disconnect' (close a session)."
)
"""Description for the mcp_client tool shown to the model."""
