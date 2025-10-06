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

import base64

from mcp.server import FastMCP
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents
from pydantic import BaseModel


class EchoResponse(BaseModel):
    """Response model for echo with structured content."""

    echoed: str
    message_length: int


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
    def echo_with_structured_content(to_echo: str) -> EchoResponse:
        return EchoResponse(echoed=to_echo, message_length=len(to_echo))

    @mcp.tool(description="Get current weather information for a location")
    def get_weather(location: str = "New York"):
        """Get weather data including forecasts and alerts for the specified location"""
        if location.lower() == "new york":
            return [
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="https://weather.api/forecast/nyc",
                        mimeType="text/plain",
                        text="Current weather in New York: 72°F, partly cloudy with light winds.",
                    ),
                )
            ]
        elif location.lower() == "london":
            return [
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="https://weather.api/data/london.json",
                        mimeType="application/json",
                        blob=base64.b64encode(
                            '{"temperature": 18, "condition": "rainy", "humidity": 85}'.encode()
                        ).decode(),
                    ),
                )
            ]
        elif location.lower() == "tokyo":
            # Simple 1x1 PNG image as base64 (weather icon)
            png_data = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )
            return [
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="https://weather.api/icons/sunny.png",
                        mimeType="image/png",
                        blob=base64.b64encode(png_data).decode(),
                    ),
                )
            ]
        else:
            return [
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"https://weather.api/forecast/{location.lower()}",
                        mimeType="text/plain",
                        text=f"Weather data for {location}: Currently unavailable.",
                    ),
                )
            ]

    mcp.run(transport="stdio")


if __name__ == "__main__":
    start_echo_server()
