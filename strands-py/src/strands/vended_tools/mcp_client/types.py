"""Shared types and constants for the mcp_client tool."""

from __future__ import annotations

from typing import Literal, TypedDict

from ...tools.mcp.mcp_client import MCPClient

Command = Literal["connect", "list_tools", "call_tool", "disconnect"]
"""Commands supported by the mcp_client tool."""


class ConnectOutput(TypedDict):
    """Output of the ``connect`` command.

    Attributes:
        session_id: Opaque identifier for the new session. Pass to subsequent commands to identify this connection.
        server: The server used to open the session.
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
    "Connects to Model Context Protocol (MCP) servers at runtime to discover and invoke their tools. "
    "'connect' opens a connection to a server and returns a session_id (the connection handle) and the "
    "server identifier. "
    "'list_tools' returns the tools the server exposes, including their names and input schemas. "
    "'call_tool' invokes a named tool on the server and returns its result. "
    "'disconnect' closes the connection. "
    "Sessions persist across turns — reuse the session_id rather than reconnecting, and disconnect when done."
)
"""Description for the mcp_client tool shown to the model."""
