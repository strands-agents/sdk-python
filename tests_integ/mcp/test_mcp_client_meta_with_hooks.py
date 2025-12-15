"""Integration test demonstrating MCP client meta field support.

This test shows how the MCP client properly handles the meta field returned by
MCP tools, both with and without structured content.
"""

from mcp import StdioServerParameters, stdio_client

from strands import Agent
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.tools.mcp.mcp_client import MCPClient


class MetaHookProvider(HookProvider):
    """Hook provider that captures tool results with meta field."""

    def __init__(self):
        self.captured_results = {}

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register callback for after tool invocation events."""
        registry.add_callback(AfterToolCallEvent, self.on_after_tool_invocation)

    def on_after_tool_invocation(self, event: AfterToolCallEvent) -> None:
        """Capture tool results."""
        tool_name = event.tool_use["name"]
        self.captured_results[tool_name] = event.result


def test_mcp_client_with_meta_only():
    """Test that MCP client correctly handles tools that return meta without structured content."""
    # Create hook provider to capture tool result
    hook_provider = MetaHookProvider()

    # Set up MCP client for echo server
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/mcp/echo_server.py"]))
    )

    with stdio_mcp_client:
        # Create agent with MCP tools and hook provider
        agent = Agent(tools=stdio_mcp_client.list_tools_sync(), hooks=[hook_provider])

        # Test meta field functionality
        test_data = "META_TEST_DATA"
        agent(f"Use the echo_with_meta tool to echo: {test_data}")

        # Verify hook captured the tool result
        assert "echo_with_meta" in hook_provider.captured_results
        result = hook_provider.captured_results["echo_with_meta"]

        # Verify basic result structure
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == test_data

        # Verify meta is present and correct
        assert "meta" in result
        assert result["meta"]["request_id"] == "test-request-123"
        assert result["meta"]["echo_length"] == len(test_data)

        # Verify structured content is not present
        assert result.get("structuredContent") is None


def test_mcp_client_with_structured_content_and_meta():
    """Test that MCP client correctly handles tools that return both structured content and meta."""
    # Create hook provider to capture tool result
    hook_provider = MetaHookProvider()

    # Set up MCP client for echo server
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/mcp/echo_server.py"]))
    )

    with stdio_mcp_client:
        # Create agent with MCP tools and hook provider
        agent = Agent(tools=stdio_mcp_client.list_tools_sync(), hooks=[hook_provider])

        # Test structured content and meta functionality
        test_data = "BOTH_TEST_DATA"
        agent(f"Use the echo_with_structured_content_and_meta tool to echo: {test_data}")

        # Verify hook captured the tool result
        assert "echo_with_structured_content_and_meta" in hook_provider.captured_results
        result = hook_provider.captured_results["echo_with_structured_content_and_meta"]

        # Verify basic result structure
        assert result["status"] == "success"
        assert len(result["content"]) == 1

        # Verify structured content is present
        assert "structuredContent" in result
        assert result["structuredContent"]["echoed"] == test_data
        assert result["structuredContent"]["message_length"] == len(test_data)

        # Verify meta is present and correct
        assert "meta" in result
        assert result["meta"]["request_id"] == "test-request-456"
        assert result["meta"]["processing_time_ms"] == 100


def test_mcp_client_without_meta():
    """Test that MCP client works correctly when tool does not return meta."""
    # Create hook provider to capture tool result
    hook_provider = MetaHookProvider()

    # Set up MCP client for echo server
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/mcp/echo_server.py"]))
    )

    with stdio_mcp_client:
        # Create agent with MCP tools and hook provider
        agent = Agent(tools=stdio_mcp_client.list_tools_sync(), hooks=[hook_provider])

        # Test regular echo tool (no meta, no structured content)
        test_data = "SIMPLE_TEST_DATA"
        agent(f"Use the echo tool to echo: {test_data}")

        # Verify hook captured the tool result
        assert "echo" in hook_provider.captured_results
        result = hook_provider.captured_results["echo"]

        # Verify basic result structure
        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == test_data

        # Verify neither meta nor structured content is present
        assert result.get("meta") is None
        assert result.get("structuredContent") is None
