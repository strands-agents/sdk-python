"""
Echo Server for MCP Integration Testing

This module implements a simple echo server using the Model Context Protocol (MCP).
It provides a basic tool that echoes back any input string, which is useful for
testing the MCP communication flow and validating that messages are properly
transmitted between the client and server.

The server runs with stdio transport, making it suitable for integration tests
where the client can spawn this process and communicate with it through standard
input/output streams.

Usage:
    Run this file directly to start the echo server:
    $ python echo_server.py
"""

from mcp.server import FastMCP


def start_echo_server():
    """
    Initialize and start the MCP echo server.

    Creates a FastMCP server instance with a single 'echo' tool that returns
    any input string back to the caller. The server uses stdio transport
    for communication.
    """
    mcp = FastMCP("Echo Server")

    @mcp.tool(description="Echos response back to the user")
    def echo(to_echo: str) -> str:
        return to_echo

    mcp.run(transport="stdio")


if __name__ == "__main__":
    start_echo_server()
