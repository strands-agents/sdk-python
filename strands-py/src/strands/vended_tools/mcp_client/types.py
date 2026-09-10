"""Shared types and constants for the mcp_client tool."""

from __future__ import annotations

from typing import Literal

Command = Literal["connect", "list_tools", "call_tool", "disconnect"]
"""Commands supported by the mcp_client tool."""


MCP_CLIENT_DESCRIPTION = (
    "Connects to Model Context Protocol (MCP) servers at runtime to discover and invoke their tools. "
    "'connect' opens a connection to a permitted server. "
    "'list_tools' returns the tools the connected server exposes, including their names and input schemas. "
    "'call_tool' invokes a named tool on the connected server and returns its result. "
    "'disconnect' closes the connection. "
    "Only one server can be connected at a time: connecting to a different server closes the current connection, "
    "and reconnecting to the same server restarts it and discards its state. Disconnect when done."
)
"""Description for the mcp_client tool shown to the model."""
