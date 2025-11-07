"""MCP server for testing elicitation.

- Docs: https://modelcontextprotocol.io/specification/draft/client/elicitation
"""

from mcp.server import FastMCP
from mcp.types import ElicitRequest, ElicitRequestParams, ElicitResult


def server() -> None:
    """Simulate approval through MCP elicitation."""
    server_ = FastMCP()

    @server_.tool(description="Tool to request approval")
    async def approval_tool() -> str:
        """Simulated approval tool.

        Returns:
            The elicitation result from the user.
        """
        request = ElicitRequest(
            params=ElicitRequestParams(
                message="Do you approve",
                requestedSchema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "request message"},
                    },
                    "required": ["message"],
                },
            ),
        )
        result = await server_.get_context().session.send_request(request, ElicitResult)

        return result.model_dump_json()

    server_.run(transport="stdio")


if __name__ == "__main__":
    server()
