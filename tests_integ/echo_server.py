"""
Echo Server for MCP Integration Testing

This module implements a simple echo server using the Model Context Protocol (MCP).
It provides basic tools that echo back input strings and structured content, which is useful for
testing the MCP communication flow and validating that messages are properly
transmitted between the client and server.

The server runs with stdio transport, making it suitable for integration tests
where the client can spawn this process and communicate with it through standard
input/output streams.

Usage:
    Run this file directly to start the echo server:
    $ python echo_server.py
"""

from typing import Any, Dict

from mcp.server import FastMCP


def start_echo_server():
    """
    Initialize and start the MCP echo server.

    Creates a FastMCP server instance with tools that return
    input strings and structured content back to the caller. The server uses stdio transport
    for communication.

    """
    mcp = FastMCP("Echo Server")

    @mcp.tool(description="Echos response back to the user", structured_output=False)
    def echo(to_echo: str) -> str:
        return to_echo

    # FastMCP automatically constructs structured output schema from method signature
    @mcp.tool(description="Echos response back with structured content", structured_output=True)
    def echo_with_structured_content(to_echo: str) -> Dict[str, Any]:
        return {"echoed": to_echo}

    mcp.run(transport="stdio")


if __name__ == "__main__":
    start_echo_server()
