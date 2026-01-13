"""Integration tests for MCP task-augmented tool execution.

These tests verify that our MCPClient correctly handles tools with taskSupport settings
and integrates with MCP servers that support task-augmented execution.

The test server (task_echo_server.py) includes a workaround for an MCP Python SDK bug
where `enable_tasks()` doesn't properly set `tasks.requests.tools.call` capability.
"""

import os
import threading
import time
from typing import Any

import pytest
from mcp.client.streamable_http import streamablehttp_client

from strands.tools.mcp.mcp_client import MCPClient
from strands.tools.mcp.mcp_types import MCPTransport


def start_task_server(port: int) -> None:
    """Start the task echo server in a thread."""
    import uvicorn

    from tests_integ.mcp.task_echo_server import create_starlette_app

    starlette_app, _ = create_starlette_app(port)
    uvicorn.run(starlette_app, host="127.0.0.1", port=port, log_level="warning")


# Use a module-level fixture port to avoid conflicts
TASK_SERVER_PORT = 8010


@pytest.fixture(scope="module")
def task_server() -> Any:
    """Start the task server for the test module."""
    server_thread = threading.Thread(target=start_task_server, kwargs={"port": TASK_SERVER_PORT}, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    yield
    # Server thread is daemon, will be cleaned up automatically


@pytest.fixture
def task_mcp_client(task_server: Any) -> MCPClient:
    """Create an MCP client connected to the task server."""

    def transport_callback() -> MCPTransport:
        return streamablehttp_client(url=f"http://127.0.0.1:{TASK_SERVER_PORT}/mcp")

    return MCPClient(transport_callback)


@pytest.mark.skipif(
    condition=os.environ.get("GITHUB_ACTIONS") == "true",
    reason="streamable transport is failing in GitHub actions",
)
class TestMCPTaskSupport:
    """Integration tests for MCP task-augmented execution.

    These tests verify our client correctly:
    1. Detects server task capability and uses task-augmented execution when appropriate
    2. Caches taskSupport settings from tools
    3. Falls back to direct call_tool for tools that don't support tasks
    4. Handles the full task workflow (call_tool_as_task -> poll_task -> get_task_result)
    """

    def test_task_forbidden_tool_uses_direct_call(self, task_mcp_client: MCPClient) -> None:
        """Test that a tool with taskSupport='forbidden' uses direct call_tool."""
        with task_mcp_client:
            # First, list tools to populate the task support cache
            tools = task_mcp_client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "task_forbidden_echo" in tool_names

            # Call the task-forbidden tool - should use direct call
            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-task-forbidden",
                name="task_forbidden_echo",
                arguments={"message": "Hello forbidden!"},
            )

            assert result["status"] == "success"
            assert len(result["content"]) == 1
            assert "Forbidden echo: Hello forbidden!" in result["content"][0].get("text", "")

    def test_tool_without_task_support_uses_direct_call(self, task_mcp_client: MCPClient) -> None:
        """Test that a tool without taskSupport setting uses direct call_tool."""
        with task_mcp_client:
            # First, list tools to populate the task support cache
            tools = task_mcp_client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "echo" in tool_names

            # Call the simple echo tool - should use direct call
            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-no-task-support",
                name="echo",
                arguments={"message": "Hello simple!"},
            )

            assert result["status"] == "success"
            assert len(result["content"]) == 1
            assert "Simple echo: Hello simple!" in result["content"][0].get("text", "")

    def test_tool_task_support_caching(self, task_mcp_client: MCPClient) -> None:
        """Test that tool taskSupport values are cached during list_tools."""
        with task_mcp_client:
            # List tools to populate the cache
            task_mcp_client.list_tools_sync()

            # Verify cache is populated
            assert task_mcp_client._get_tool_task_support("task_required_echo") == "required"
            assert task_mcp_client._get_tool_task_support("task_optional_echo") == "optional"
            assert task_mcp_client._get_tool_task_support("task_forbidden_echo") == "forbidden"
            # Tool without explicit setting should have None in cache
            assert task_mcp_client._get_tool_task_support("echo") is None

    def test_server_capabilities_advertised(self, task_mcp_client: MCPClient) -> None:
        """Test that server properly advertises task capabilities."""
        with task_mcp_client:
            # List tools first to initialize the connection
            task_mcp_client.list_tools_sync()

            # Get raw capabilities to verify structure
            session = task_mcp_client._background_thread_session
            if session:
                caps = session.get_server_capabilities()
                assert caps is not None
                assert caps.tasks is not None
                assert caps.tasks.requests is not None
                assert caps.tasks.requests.tools is not None
                # Server properly advertises call capability (via our workaround)
                assert caps.tasks.requests.tools.call is not None

            # Our capability check correctly returns True
            assert task_mcp_client._has_server_task_support() is True

    def test_task_required_tool_uses_task_execution(self, task_mcp_client: MCPClient) -> None:
        """Test that task-required tools use task-augmented execution."""
        with task_mcp_client:
            tools = task_mcp_client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "task_required_echo" in tool_names

            # Tool is marked as task-required and server advertises capability
            # So we use task-augmented execution (call_tool_as_task -> poll -> get_result)
            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-task-required",
                name="task_required_echo",
                arguments={"message": "Hello from task!"},
            )

            assert result["status"] == "success"
            assert len(result["content"]) == 1
            assert "Task echo: Hello from task!" in result["content"][0].get("text", "")

    def test_task_optional_tool_uses_task_execution(self, task_mcp_client: MCPClient) -> None:
        """Test that task-optional tools use task-augmented execution when server supports it."""
        with task_mcp_client:
            tools = task_mcp_client.list_tools_sync()
            tool_names = [t.tool_name for t in tools]
            assert "task_optional_echo" in tool_names

            # Tool is marked as task-optional and server advertises capability
            # So we prefer task-augmented execution
            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-task-optional",
                name="task_optional_echo",
                arguments={"message": "Hello optional task!"},
            )

            assert result["status"] == "success"
            assert len(result["content"]) == 1
            # Tool returns "Task optional echo" when in task mode
            assert "Task optional echo: Hello optional task!" in result["content"][0].get("text", "")

    def test_should_use_task_logic_with_server_support(self, task_mcp_client: MCPClient) -> None:
        """Test that _should_use_task returns correct values based on tool taskSupport."""
        with task_mcp_client:
            # List tools to populate caches
            task_mcp_client.list_tools_sync()

            # Server advertises tasks.requests.tools.call, so task-required and task-optional
            # should use tasks, while forbidden and unset should not
            assert task_mcp_client._should_use_task("task_required_echo") is True
            assert task_mcp_client._should_use_task("task_optional_echo") is True
            assert task_mcp_client._should_use_task("task_forbidden_echo") is False
            assert task_mcp_client._should_use_task("echo") is False

    def test_multiple_tool_calls_in_sequence(self, task_mcp_client: MCPClient) -> None:
        """Test calling multiple tools in sequence with different task modes."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()

            # Call task-forbidden tool (uses direct call)
            result1 = task_mcp_client.call_tool_sync(
                tool_use_id="seq-1",
                name="task_forbidden_echo",
                arguments={"message": "First"},
            )
            assert result1["status"] == "success"
            assert "Forbidden echo: First" in result1["content"][0].get("text", "")

            # Call simple echo (uses direct call - no taskSupport setting)
            result2 = task_mcp_client.call_tool_sync(
                tool_use_id="seq-2",
                name="echo",
                arguments={"message": "Second"},
            )
            assert result2["status"] == "success"
            assert "Simple echo: Second" in result2["content"][0].get("text", "")

            # Call task-optional (uses task execution since server supports it)
            result3 = task_mcp_client.call_tool_sync(
                tool_use_id="seq-3",
                name="task_optional_echo",
                arguments={"message": "Third"},
            )
            assert result3["status"] == "success"
            assert "Task optional echo: Third" in result3["content"][0].get("text", "")

            # Call task-required (uses task execution)
            result4 = task_mcp_client.call_tool_sync(
                tool_use_id="seq-4",
                name="task_required_echo",
                arguments={"message": "Fourth"},
            )
            assert result4["status"] == "success"
            assert "Task echo: Fourth" in result4["content"][0].get("text", "")

    @pytest.mark.asyncio
    async def test_async_tool_calls(self, task_mcp_client: MCPClient) -> None:
        """Test async tool calls work correctly."""
        with task_mcp_client:
            # List tools first
            task_mcp_client.list_tools_sync()

            # Call tool asynchronously
            result = await task_mcp_client.call_tool_async(
                tool_use_id="test-async",
                name="task_forbidden_echo",
                arguments={"message": "Async hello!"},
            )

            assert result["status"] == "success"
            assert "Forbidden echo: Async hello!" in result["content"][0].get("text", "")
