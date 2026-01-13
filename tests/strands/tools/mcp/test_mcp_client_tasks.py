"""Tests for MCP task-augmented execution support in MCPClient.

These unit tests focus on error handling and edge cases that are not easily
testable through integration tests. Happy-path flows are covered by
integration tests in tests_integ/mcp/test_mcp_client_tasks.py.
"""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ListToolsResult
from mcp.types import Tool as MCPTool
from mcp.types import ToolExecution

from strands.tools.mcp import MCPClient

from .conftest import create_server_capabilities


class TestTaskExecutionFailures:
    """Tests for task execution failure handling."""

    @pytest.mark.parametrize(
        "status,status_message,expected_text",
        [
            ("failed", "Something went wrong", "Something went wrong"),
            ("cancelled", None, "cancelled"),
        ],
    )
    def test_task_execution_terminal_status(self, mock_transport, mock_session, status, status_message, expected_text):
        """Test handling of terminal task statuses (failed, cancelled)."""
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = f"task-{status}"
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        mock_status = MagicMock()
        mock_status.status = status
        mock_status.statusMessage = status_message

        async def mock_poll_task(task_id):
            yield mock_status

        mock_session.experimental.poll_task = mock_poll_task

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["test_tool"] = "required"
            result = client.call_tool_sync(tool_use_id="test-id", name="test_tool", arguments={})

            assert result["status"] == "error"
            assert expected_text.lower() in result["content"][0].get("text", "").lower()


class TestStopResetCache:
    """Tests for cache reset in stop()."""

    def test_stop_resets_task_caches(self, mock_transport, mock_session):
        """Test that stop() resets the task support caches."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["tool1"] = "required"

        assert client._server_task_capable is None
        assert client._tool_task_support_cache == {}


class TestTaskConfiguration:
    """Tests for task-related configuration options."""

    def test_default_task_config_values(self, mock_transport, mock_session):
        """Test default configuration values."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._default_task_ttl_ms == 60000
            assert client._default_task_poll_timeout_seconds == 300.0

    def test_custom_task_config_values(self, mock_transport, mock_session):
        """Test custom configuration values."""
        with MCPClient(
            mock_transport["transport_callable"],
            default_task_ttl_ms=120000,
            default_task_poll_timeout_seconds=60.0,
        ) as client:
            assert client._default_task_ttl_ms == 120000
            assert client._default_task_poll_timeout_seconds == 60.0


class TestTaskExecutionTimeout:
    """Tests for task execution timeout and error handling."""

    def _setup_task_tool(self, mock_session, tool_name: str) -> None:
        """Helper to set up a mock task-enabled tool."""
        mock_session.get_server_capabilities = MagicMock(return_value=create_server_capabilities(True))
        mock_tool = MCPTool(
            name=tool_name,
            description="A test tool",
            inputSchema={"type": "object"},
            execution=ToolExecution(taskSupport="optional"),
        )
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[mock_tool], nextCursor=None))

        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "test-task-id"
        mock_session.experimental = MagicMock()
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

    @pytest.mark.asyncio
    async def test_task_polling_timeout(self, mock_transport, mock_session):
        """Test that task polling times out properly."""
        self._setup_task_tool(mock_session, "slow_tool")

        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        with MCPClient(mock_transport["transport_callable"], default_task_poll_timeout_seconds=0.1) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="test-123", name="slow_tool", arguments={})

            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_result_retrieval_failure(self, mock_transport, mock_session):
        """Test that get_task_result failures are handled gracefully."""
        self._setup_task_tool(mock_session, "failing_tool")

        async def successful_poll(task_id):
            yield MagicMock(status="completed", statusMessage=None)

        mock_session.experimental.poll_task = successful_poll
        mock_session.experimental.get_task_result = AsyncMock(side_effect=Exception("Network error"))

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="test-456", name="failing_tool", arguments={})

            assert result["status"] == "error"
            assert "result retrieval failed" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_explicit_timeout_overrides_default(self, mock_transport, mock_session):
        """Test that read_timeout_seconds overrides the default poll timeout."""
        self._setup_task_tool(mock_session, "timeout_tool")

        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        # Long default timeout, but short explicit timeout
        with MCPClient(mock_transport["transport_callable"], default_task_poll_timeout_seconds=300.0) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(
                tool_use_id="test-timeout",
                name="timeout_tool",
                arguments={},
                read_timeout_seconds=timedelta(seconds=0.1),
            )

            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_polling_yields_no_status(self, mock_transport, mock_session):
        """Test handling when poll_task yields nothing (final_status is None)."""
        self._setup_task_tool(mock_session, "empty_poll_tool")

        async def empty_poll(task_id):
            return
            yield  # noqa: B901 - makes this an async generator

        mock_session.experimental.poll_task = empty_poll

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="empty_poll_tool", arguments={})
            assert result["status"] == "error"
            assert "without status" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_unexpected_terminal_status(self, mock_transport, mock_session):
        """Test handling of unexpected task status (not completed/failed/cancelled)."""
        self._setup_task_tool(mock_session, "weird_tool")

        async def poll(task_id):
            yield MagicMock(status="unknown_status", statusMessage=None)

        mock_session.experimental.poll_task = poll

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="weird_tool", arguments={})
            assert result["status"] == "error"
            assert "unexpected task status" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_successful_completion(self, mock_transport, mock_session):
        """Test successful task completion with result retrieval (happy path)."""
        from mcp.types import CallToolResult as MCPCallToolResult
        from mcp.types import TextContent as MCPTextContent

        self._setup_task_tool(mock_session, "success_tool")

        async def poll(task_id):
            yield MagicMock(status="completed", statusMessage=None)

        mock_session.experimental.poll_task = poll
        mock_session.experimental.get_task_result = AsyncMock(
            return_value=MCPCallToolResult(content=[MCPTextContent(type="text", text="Done")], isError=False)
        )

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="success_tool", arguments={})
            assert result["status"] == "success"
            assert "Done" in result["content"][0].get("text", "")
