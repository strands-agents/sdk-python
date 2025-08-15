"""Integration test demonstrating hooks system with MCP client structured content tool.

This test shows how to use the hooks system to capture and inspect tool invocation
results, specifically testing the echo_with_structured_content tool from echo_server.
"""

import json

from mcp import StdioServerParameters, stdio_client

from strands import Agent
from strands.experimental.hooks import AfterToolInvocationEvent
from strands.hooks import HookProvider, HookRegistry
from strands.tools.mcp.mcp_client import MCPClient


class StructuredContentHookProvider(HookProvider):
    """Hook provider that captures structured content tool results."""

    def __init__(self):
        self.captured_result = None

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register callback for after tool invocation events."""
        registry.add_callback(AfterToolInvocationEvent, self.on_after_tool_invocation)

    def on_after_tool_invocation(self, event: AfterToolInvocationEvent) -> None:
        """Capture structured content tool results."""
        if event.tool_use["name"] == "echo_with_structured_content":
            self.captured_result = event.result


def test_mcp_client_hooks_structured_content():
    """Test using hooks to inspect echo_with_structured_content tool result."""
    # Create hook provider to capture tool result
    hook_provider = StructuredContentHookProvider()

    # Set up MCP client for echo server
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/echo_server.py"]))
    )

    with stdio_mcp_client:
        # Create agent with MCP tools and hook provider
        agent = Agent(tools=stdio_mcp_client.list_tools_sync(), hooks=[hook_provider])

        # Test structured content functionality
        test_data = "HOOKS_TEST_DATA"
        agent(f"Use the echo_with_structured_content tool to echo: {test_data}")

        # Verify hook captured the tool result
        assert hook_provider.captured_result is not None
        result = hook_provider.captured_result

        # Verify basic result structure
        assert result["status"] == "success"
        assert len(result["content"]) == 1

        # Verify structured content is present and correct
        assert "structuredContent" in result
        assert result["structuredContent"]["result"] == {"echoed": test_data}

        # Verify text content matches structured content
        text_content = json.loads(result["content"][0]["text"])
        assert text_content == {"echoed": test_data}
