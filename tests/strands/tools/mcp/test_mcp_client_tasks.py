"""Tests for MCP task-augmented execution support in MCPClient."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ListToolsResult
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool
from mcp.types import ToolExecution

from strands.tools.mcp import MCPClient
from strands.tools.mcp.mcp_client import DEFAULT_TASK_POLL_TIMEOUT, DEFAULT_TASK_TTL

from .conftest import create_server_capabilities


class TestTasksOptIn:
    """Tests for task opt-in behavior via experimental.tasks."""

    @pytest.mark.parametrize(
        "experimental,expected_enabled",
        [
            (None, False),
            ({}, False),
            ({"tasks": None}, False),
            ({"tasks": {}}, True),
            ({"tasks": {"ttl_ms": 1000}}, True),
        ],
    )
    def test_tasks_enabled_state(self, mock_transport, mock_session, experimental, expected_enabled):
        """Test _is_tasks_enabled based on experimental config."""
        with MCPClient(mock_transport["transport_callable"], experimental=experimental) as client:
            assert client._is_tasks_enabled() is expected_enabled

    def test_should_use_task_requires_opt_in(self, mock_transport, mock_session):
        """Test that _should_use_task returns False without opt-in even with server/tool support."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            assert client._should_use_task("test_tool") is False

        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["test_tool"] = "required"
            assert client._should_use_task("test_tool") is True


class TestTaskConfiguration:
    """Tests for task-related configuration options."""

    @pytest.mark.parametrize(
        "config,expected_ttl,expected_timeout",
        [
            ({}, DEFAULT_TASK_TTL, DEFAULT_TASK_POLL_TIMEOUT),
            ({"ttl": 120000}, 120000, DEFAULT_TASK_POLL_TIMEOUT),
            ({"poll_timeout": 60.0}, DEFAULT_TASK_TTL, 60.0),
            ({"ttl": 120000, "poll_timeout": 60.0}, 120000, 60.0),
        ],
    )
    def test_task_config_values(self, mock_transport, mock_session, config, expected_ttl, expected_timeout):
        """Test task configuration values with various configs."""
        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": config}) as client:
            config_actual = client._get_task_config()
            assert config_actual.get("ttl") == expected_ttl
            assert config_actual.get("poll_timeout") == expected_timeout

    def test_stop_resets_task_caches(self, mock_transport, mock_session):
        """Test that stop() resets the task support caches."""
        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["tool1"] = "required"
        assert client._server_task_capable is None
        assert client._tool_task_support_cache == {}


class TestTaskExecution:
    """Tests for task execution and error handling."""

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

    @pytest.mark.parametrize(
        "status,status_message,expected_text",
        [
            ("failed", "Something went wrong", "Something went wrong"),
            ("cancelled", None, "cancelled"),
            ("unknown_status", None, "unexpected task status"),
        ],
    )
    def test_terminal_status_handling(self, mock_transport, mock_session, status, status_message, expected_text):
        """Test handling of terminal task statuses."""
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = f"task-{status}"
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        async def mock_poll_task(task_id):
            yield MagicMock(status=status, statusMessage=status_message)

        mock_session.experimental.poll_task = mock_poll_task

        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["test_tool"] = "required"
            result = client.call_tool_sync(tool_use_id="test-id", name="test_tool", arguments={})
            assert result["status"] == "error"
            assert expected_text.lower() in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_polling_timeout(self, mock_transport, mock_session):
        """Test that task polling times out properly."""
        self._setup_task_tool(mock_session, "slow_tool")

        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        with MCPClient(
            mock_transport["transport_callable"], experimental={"tasks": {"poll_timeout": timedelta(seconds=0.1)}}
        ) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="slow_tool", arguments={})
            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_explicit_timeout_overrides_default(self, mock_transport, mock_session):
        """Test that read_timeout_seconds overrides the default poll timeout."""
        self._setup_task_tool(mock_session, "timeout_tool")

        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        with MCPClient(
            mock_transport["transport_callable"], experimental={"tasks": {"poll_timeout": timedelta(minutes=5)}}
        ) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(
                tool_use_id="t", name="timeout_tool", arguments={}, read_timeout_seconds=timedelta(seconds=0.1)
            )
            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_result_retrieval_failure(self, mock_transport, mock_session):
        """Test that get_task_result failures are handled gracefully."""
        self._setup_task_tool(mock_session, "failing_tool")

        async def successful_poll(task_id):
            yield MagicMock(status="completed", statusMessage=None)

        mock_session.experimental.poll_task = successful_poll
        mock_session.experimental.get_task_result = AsyncMock(side_effect=Exception("Network error"))

        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="failing_tool", arguments={})
            assert result["status"] == "error"
            assert "result retrieval failed" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_empty_poll_result(self, mock_transport, mock_session):
        """Test handling when poll_task yields nothing."""
        self._setup_task_tool(mock_session, "empty_poll_tool")

        async def empty_poll(task_id):
            return
            yield  # noqa: B901

        mock_session.experimental.poll_task = empty_poll

        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="empty_poll_tool", arguments={})
            assert result["status"] == "error"
            assert "without status" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_transport, mock_session):
        """Test successful task completion."""
        self._setup_task_tool(mock_session, "success_tool")

        async def poll(task_id):
            yield MagicMock(status="completed", statusMessage=None)

        mock_session.experimental.poll_task = poll
        mock_session.experimental.get_task_result = AsyncMock(
            return_value=MCPCallToolResult(content=[MCPTextContent(type="text", text="Done")], isError=False)
        )

        with MCPClient(mock_transport["transport_callable"], experimental={"tasks": {}}) as client:
            client.list_tools_sync()
            result = await client.call_tool_async(tool_use_id="t", name="success_tool", arguments={})
            assert result["status"] == "success"
            assert "Done" in result["content"][0].get("text", "")
