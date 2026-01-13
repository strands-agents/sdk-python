"""Integration tests for MCP task-augmented tool execution.

These tests verify that our MCPClient correctly handles tools with taskSupport settings
and integrates with MCP servers that support task-augmented execution.

The test server (task_echo_server.py) includes a workaround for an MCP Python SDK bug
where `enable_tasks()` doesn't properly set `tasks.requests.tools.call` capability.
"""

import os
import socket
import threading
import time
from typing import Any

import pytest
from mcp.client.streamable_http import streamablehttp_client

from strands.tools.mcp.mcp_client import MCPClient
from strands.tools.mcp.mcp_types import MCPTransport


def _find_available_port() -> int:
    """Find an available port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_task_server(port: int) -> None:
    """Start the task echo server in a thread."""
    import uvicorn

    from tests_integ.mcp.task_echo_server import create_starlette_app

    starlette_app, _ = create_starlette_app(port)
    uvicorn.run(starlette_app, host="127.0.0.1", port=port, log_level="warning")


@pytest.fixture(scope="module")
def task_server_port() -> int:
    """Get a dynamically allocated port for the task server."""
    return _find_available_port()


@pytest.fixture(scope="module")
def task_server(task_server_port: int) -> Any:
    """Start the task server for the test module."""
    server_thread = threading.Thread(target=start_task_server, kwargs={"port": task_server_port}, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    yield
    # Server thread is daemon, will be cleaned up automatically


@pytest.fixture
def task_mcp_client(task_server: Any, task_server_port: int) -> MCPClient:
    """Create an MCP client connected to the task server."""

    def transport_callback() -> MCPTransport:
        return streamablehttp_client(url=f"http://127.0.0.1:{task_server_port}/mcp")

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
            tools = task_mcp_client.list_tools_sync()
            assert "task_forbidden_echo" in [t.tool_name for t in tools]

            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-1", name="task_forbidden_echo", arguments={"message": "Hello forbidden!"}
            )
            assert result["status"] == "success"
            assert "Forbidden echo: Hello forbidden!" in result["content"][0].get("text", "")

    def test_tool_without_task_support_uses_direct_call(self, task_mcp_client: MCPClient) -> None:
        """Test that a tool without taskSupport setting uses direct call_tool."""
        with task_mcp_client:
            tools = task_mcp_client.list_tools_sync()
            assert "echo" in [t.tool_name for t in tools]

            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-2", name="echo", arguments={"message": "Hello simple!"}
            )
            assert result["status"] == "success"
            assert "Simple echo: Hello simple!" in result["content"][0].get("text", "")

    def test_tool_task_support_caching(self, task_mcp_client: MCPClient) -> None:
        """Test that tool taskSupport values are cached during list_tools."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()
            assert task_mcp_client._get_tool_task_support("task_required_echo") == "required"
            assert task_mcp_client._get_tool_task_support("task_optional_echo") == "optional"
            assert task_mcp_client._get_tool_task_support("task_forbidden_echo") == "forbidden"
            assert task_mcp_client._get_tool_task_support("echo") is None

    def test_server_capabilities_advertised(self, task_mcp_client: MCPClient) -> None:
        """Test that server properly advertises task capabilities."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()
            session = task_mcp_client._background_thread_session
            if session:
                caps = session.get_server_capabilities()
                assert caps is not None and caps.tasks is not None
                assert caps.tasks.requests is not None and caps.tasks.requests.tools is not None
                assert caps.tasks.requests.tools.call is not None
            assert task_mcp_client._has_server_task_support() is True

    def test_task_required_tool_uses_task_execution(self, task_mcp_client: MCPClient) -> None:
        """Test that task-required tools use task-augmented execution."""
        with task_mcp_client:
            tools = task_mcp_client.list_tools_sync()
            assert "task_required_echo" in [t.tool_name for t in tools]

            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-3", name="task_required_echo", arguments={"message": "Hello from task!"}
            )
            assert result["status"] == "success"
            assert "Task echo: Hello from task!" in result["content"][0].get("text", "")

    def test_task_optional_tool_uses_task_execution(self, task_mcp_client: MCPClient) -> None:
        """Test that task-optional tools use task-augmented execution when server supports it."""
        with task_mcp_client:
            tools = task_mcp_client.list_tools_sync()
            assert "task_optional_echo" in [t.tool_name for t in tools]

            result = task_mcp_client.call_tool_sync(
                tool_use_id="test-4", name="task_optional_echo", arguments={"message": "Hello optional task!"}
            )
            assert result["status"] == "success"
            assert "Task optional echo: Hello optional task!" in result["content"][0].get("text", "")

    def test_should_use_task_logic_with_server_support(self, task_mcp_client: MCPClient) -> None:
        """Test that _should_use_task returns correct values based on tool taskSupport."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()
            assert task_mcp_client._should_use_task("task_required_echo") is True
            assert task_mcp_client._should_use_task("task_optional_echo") is True
            assert task_mcp_client._should_use_task("task_forbidden_echo") is False
            assert task_mcp_client._should_use_task("echo") is False

    def test_multiple_tool_calls_in_sequence(self, task_mcp_client: MCPClient) -> None:
        """Test calling multiple tools in sequence with different task modes."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()

            r1 = task_mcp_client.call_tool_sync(
                tool_use_id="s1", name="task_forbidden_echo", arguments={"message": "1"}
            )
            assert r1["status"] == "success" and "Forbidden echo: 1" in r1["content"][0].get("text", "")

            r2 = task_mcp_client.call_tool_sync(tool_use_id="s2", name="echo", arguments={"message": "2"})
            assert r2["status"] == "success" and "Simple echo: 2" in r2["content"][0].get("text", "")

            r3 = task_mcp_client.call_tool_sync(tool_use_id="s3", name="task_optional_echo", arguments={"message": "3"})
            assert r3["status"] == "success" and "Task optional echo: 3" in r3["content"][0].get("text", "")

            r4 = task_mcp_client.call_tool_sync(tool_use_id="s4", name="task_required_echo", arguments={"message": "4"})
            assert r4["status"] == "success" and "Task echo: 4" in r4["content"][0].get("text", "")

    @pytest.mark.asyncio
    async def test_async_tool_calls(self, task_mcp_client: MCPClient) -> None:
        """Test async tool calls work correctly."""
        with task_mcp_client:
            task_mcp_client.list_tools_sync()
            result = await task_mcp_client.call_tool_async(
                tool_use_id="test-async", name="task_forbidden_echo", arguments={"message": "Async hello!"}
            )
            assert result["status"] == "success"
            assert "Forbidden echo: Async hello!" in result["content"][0].get("text", "")
