"""Tests for MCP task-augmented execution support in MCPClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ListToolsResult
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import TaskExecutionMode, ToolExecution
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool

from strands.tools.mcp import MCPClient


@pytest.fixture
def mock_transport():
    """Create a mock MCP transport."""
    mock_read_stream = AsyncMock()
    mock_write_stream = AsyncMock()
    mock_transport_cm = AsyncMock()
    mock_transport_cm.__aenter__.return_value = (mock_read_stream, mock_write_stream)
    mock_transport_callable = MagicMock(return_value=mock_transport_cm)

    return {
        "read_stream": mock_read_stream,
        "write_stream": mock_write_stream,
        "transport_cm": mock_transport_cm,
        "transport_callable": mock_transport_callable,
    }


@pytest.fixture
def mock_session():
    """Create a mock MCP session."""
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    # Default: no task support (get_server_capabilities is sync, not async!)
    mock_session.get_server_capabilities = MagicMock(return_value=None)

    # Create a mock context manager for ClientSession
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session

    # Patch ClientSession to return our mock session
    with patch("strands.tools.mcp.mcp_client.ClientSession", return_value=mock_session_cm):
        yield mock_session


def _create_server_capabilities(has_task_support: bool):
    """Create mock server capabilities."""
    caps = MagicMock()
    if has_task_support:
        caps.tasks = MagicMock()
        caps.tasks.requests = MagicMock()
        caps.tasks.requests.tools = MagicMock()
        caps.tasks.requests.tools.call = MagicMock()
    else:
        caps.tasks = None
    return caps


def _create_tool_with_task_support(name: str, task_support: TaskExecutionMode | None):
    """Create a mock MCPTool with the specified taskSupport."""
    if task_support is not None:
        execution = ToolExecution(taskSupport=task_support)
    else:
        execution = None
    return MCPTool(
        name=name,
        description=f"Tool {name}",
        inputSchema={"type": "object", "properties": {}},
        execution=execution,
    )


class TestHasServerTaskSupport:
    """Tests for _has_server_task_support() method.

    Note: _has_server_task_support() returns the cached value set during list_tools_sync().
    If the cache hasn't been populated, it returns False (conservative default).
    """

    def test_has_server_task_support_returns_cached_true(self, mock_transport, mock_session):
        """Test that _has_server_task_support returns True when cache is True."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            assert client._has_server_task_support() is True

    def test_has_server_task_support_returns_cached_false(self, mock_transport, mock_session):
        """Test that _has_server_task_support returns False when cache is False."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = False
            assert client._has_server_task_support() is False

    def test_has_server_task_support_returns_false_when_not_cached(self, mock_transport, mock_session):
        """Test that _has_server_task_support returns False when cache is None (not yet populated)."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            # Cache is None by default
            assert client._server_task_capable is None
            assert client._has_server_task_support() is False


class TestGetToolTaskSupport:
    """Tests for _get_tool_task_support() method."""

    def test_get_tool_task_support_cached(self, mock_transport, mock_session):
        """Test that _get_tool_task_support returns cached value."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._tool_task_support_cache["my_tool"] = "required"
            result = client._get_tool_task_support("my_tool")
            assert result == "required"

    def test_get_tool_task_support_not_cached(self, mock_transport, mock_session):
        """Test that _get_tool_task_support returns None for uncached tool."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            result = client._get_tool_task_support("unknown_tool")
            assert result is None


class TestShouldUseTask:
    """Tests for _should_use_task() method."""

    def test_should_use_task_required_server_supports(self, mock_transport, mock_session):
        """Test that _should_use_task returns True for required taskSupport when server supports."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["my_tool"] = "required"
            result = client._should_use_task("my_tool")
            assert result is True

    def test_should_use_task_required_server_no_support(self, mock_transport, mock_session):
        """Test that _should_use_task returns False for required when server doesn't support (per MCP spec)."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = False
            client._tool_task_support_cache["my_tool"] = "required"
            result = client._should_use_task("my_tool")
            # Per MCP spec, client MUST NOT use tasks when server doesn't support
            assert result is False

    def test_should_use_task_optional_server_supports(self, mock_transport, mock_session):
        """Test that _should_use_task returns True for optional taskSupport when server supports."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["my_tool"] = "optional"
            result = client._should_use_task("my_tool")
            assert result is True

    def test_should_use_task_optional_server_no_support(self, mock_transport, mock_session):
        """Test that _should_use_task returns False for optional when server doesn't support."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = False
            client._tool_task_support_cache["my_tool"] = "optional"
            result = client._should_use_task("my_tool")
            assert result is False

    def test_should_use_task_forbidden(self, mock_transport, mock_session):
        """Test that _should_use_task returns False for forbidden taskSupport."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["my_tool"] = "forbidden"
            result = client._should_use_task("my_tool")
            assert result is False

    def test_should_use_task_none(self, mock_transport, mock_session):
        """Test that _should_use_task returns False when taskSupport is None."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["my_tool"] = None
            result = client._should_use_task("my_tool")
            assert result is False

    def test_should_use_task_uncached_tool(self, mock_transport, mock_session):
        """Test that _should_use_task returns False for uncached tool (defaults to forbidden)."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            result = client._should_use_task("uncached_tool")
            assert result is False


class TestListToolsSyncTaskCaching:
    """Tests for taskSupport caching in list_tools_sync()."""

    def test_list_tools_sync_caches_task_support(self, mock_transport, mock_session):
        """Test that list_tools_sync caches taskSupport for each tool."""
        tools = [
            _create_tool_with_task_support("required_tool", "required"),
            _create_tool_with_task_support("optional_tool", "optional"),
            _create_tool_with_task_support("forbidden_tool", "forbidden"),
            _create_tool_with_task_support("no_support_tool", None),
        ]
        mock_session.list_tools.return_value = ListToolsResult(tools=tools)

        with MCPClient(mock_transport["transport_callable"]) as client:
            client.list_tools_sync()

            assert client._tool_task_support_cache["required_tool"] == "required"
            assert client._tool_task_support_cache["optional_tool"] == "optional"
            assert client._tool_task_support_cache["forbidden_tool"] == "forbidden"
            assert client._tool_task_support_cache["no_support_tool"] is None

    def test_list_tools_sync_caches_server_task_capability_true(self, mock_transport, mock_session):
        """Test that list_tools_sync caches server task capability when supported."""
        mock_session.list_tools.return_value = ListToolsResult(tools=[])
        mock_session.get_server_capabilities = MagicMock(
            return_value=_create_server_capabilities(has_task_support=True)
        )

        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._server_task_capable is None
            client.list_tools_sync()
            assert client._server_task_capable is True

    def test_list_tools_sync_caches_server_task_capability_false(self, mock_transport, mock_session):
        """Test that list_tools_sync caches server task capability when not supported."""
        mock_session.list_tools.return_value = ListToolsResult(tools=[])
        mock_session.get_server_capabilities = MagicMock(
            return_value=_create_server_capabilities(has_task_support=False)
        )

        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._server_task_capable is None
            client.list_tools_sync()
            assert client._server_task_capable is False

    def test_list_tools_sync_caches_server_task_capability_none(self, mock_transport, mock_session):
        """Test that list_tools_sync caches False when capabilities are None."""
        mock_session.list_tools.return_value = ListToolsResult(tools=[])
        mock_session.get_server_capabilities = MagicMock(return_value=None)

        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._server_task_capable is None
            client.list_tools_sync()
            assert client._server_task_capable is False


class TestCallToolSyncWithTasks:
    """Tests for call_tool_sync with task support."""

    def test_call_tool_sync_uses_task_when_appropriate(self, mock_transport, mock_session):
        """Test that call_tool_sync uses task-augmented execution when appropriate."""
        # Mock the experimental task methods
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "task-123"
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task as an async generator
        mock_status = MagicMock()
        mock_status.status = "completed"

        async def mock_poll_task(task_id):
            yield mock_status

        mock_session.experimental.poll_task = mock_poll_task

        # Mock get_task_result
        mock_result = MCPCallToolResult(
            isError=False,
            content=[MCPTextContent(type="text", text="Task completed")],
        )
        mock_session.experimental.get_task_result = AsyncMock(return_value=mock_result)

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["task_tool"] = "required"
            result = client.call_tool_sync(tool_use_id="test-123", name="task_tool", arguments={"param": "value"})

            # Verify task-augmented execution was used
            mock_session.experimental.call_tool_as_task.assert_called_once()
            mock_session.experimental.get_task_result.assert_called_once()

            assert result["status"] == "success"
            assert result["toolUseId"] == "test-123"

    def test_call_tool_sync_uses_direct_call_when_no_task_support(self, mock_transport, mock_session):
        """Test that call_tool_sync uses direct call when tool doesn't support tasks."""
        mock_session.call_tool.return_value = MCPCallToolResult(
            isError=False,
            content=[MCPTextContent(type="text", text="Direct result")],
        )

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["direct_tool"] = "forbidden"
            result = client.call_tool_sync(tool_use_id="test-123", name="direct_tool", arguments={"param": "value"})

            # Verify direct call was used
            mock_session.call_tool.assert_called_once()

            assert result["status"] == "success"
            assert result["toolUseId"] == "test-123"


class TestCallToolAsyncWithTasks:
    """Tests for call_tool_async with task support."""

    @pytest.mark.asyncio
    async def test_call_tool_async_uses_task_when_appropriate(self, mock_transport, mock_session):
        """Test that call_tool_async uses task-augmented execution when appropriate."""
        from concurrent import futures

        # Mock result to return
        mock_result = MCPCallToolResult(
            isError=False,
            content=[MCPTextContent(type="text", text="Async task completed")],
        )

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["async_task_tool"] = "optional"

            # Mock _invoke_on_background_thread to avoid creating orphaned coroutines
            # This approach is cleaner than mocking asyncio functions because the inner
            # coroutine (_call_as_task) is never created when the mock intercepts the call
            captured_coros: list = []

            def mock_invoke(coro):
                # Capture and close the coroutine to prevent warnings
                captured_coros.append(coro)
                coro.close()
                # Return a completed future with the mock result
                future: futures.Future = futures.Future()
                future.set_result(mock_result)
                return future

            with patch.object(client, "_invoke_on_background_thread", side_effect=mock_invoke):
                result = await client.call_tool_async(
                    tool_use_id="test-456", name="async_task_tool", arguments={"param": "value"}
                )

            # Verify task-augmented path was taken (the coroutine name tells us)
            assert len(captured_coros) == 1
            assert "_call_as_task" in captured_coros[0].__qualname__

            assert result["status"] == "success"
            assert result["toolUseId"] == "test-456"


class TestTaskExecutionFailures:
    """Tests for task execution failure handling."""

    def test_task_execution_failed_status(self, mock_transport, mock_session):
        """Test handling of failed task status."""
        # Mock the experimental task methods
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "task-failed"
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task returning failed status
        mock_status = MagicMock()
        mock_status.status = "failed"
        mock_status.statusMessage = "Something went wrong"

        async def mock_poll_task(task_id):
            yield mock_status

        mock_session.experimental.poll_task = mock_poll_task

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["failing_tool"] = "required"
            result = client.call_tool_sync(tool_use_id="test-fail", name="failing_tool", arguments={})

            # Should return error result
            assert result["status"] == "error"
            assert "Something went wrong" in result["content"][0].get("text", "")

    def test_task_execution_cancelled_status(self, mock_transport, mock_session):
        """Test handling of cancelled task status."""
        # Mock the experimental task methods
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "task-cancelled"
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task returning cancelled status
        mock_status = MagicMock()
        mock_status.status = "cancelled"

        async def mock_poll_task(task_id):
            yield mock_status

        mock_session.experimental.poll_task = mock_poll_task

        with MCPClient(mock_transport["transport_callable"]) as client:
            client._server_task_capable = True
            client._tool_task_support_cache["cancelled_tool"] = "required"
            result = client.call_tool_sync(tool_use_id="test-cancel", name="cancelled_tool", arguments={})

            # Should return error result
            assert result["status"] == "error"
            assert "cancelled" in result["content"][0].get("text", "").lower()


class TestStopResetCache:
    """Tests for cache reset in stop()."""

    def test_stop_resets_task_caches(self, mock_transport, mock_session):
        """Test that stop() resets the task support caches."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            # Populate caches
            client._server_task_capable = True
            client._tool_task_support_cache["tool1"] = "required"
            client._tool_task_support_cache["tool2"] = "optional"

        # After exiting context, caches should be reset
        assert client._server_task_capable is None
        assert client._tool_task_support_cache == {}


class TestDefaultTaskTtl:
    """Tests for default_task_ttl_ms configuration."""

    def test_default_task_ttl_ms_default_value(self, mock_transport, mock_session):
        """Test that default_task_ttl_ms has correct default value."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._default_task_ttl_ms == 60000

    def test_default_task_ttl_ms_custom_value(self, mock_transport, mock_session):
        """Test that default_task_ttl_ms can be customized."""
        with MCPClient(mock_transport["transport_callable"], default_task_ttl_ms=120000) as client:
            assert client._default_task_ttl_ms == 120000


class TestDefaultTaskPollTimeout:
    """Tests for default_task_poll_timeout_seconds configuration."""

    def test_default_task_poll_timeout_default_value(self, mock_transport, mock_session):
        """Test that default_task_poll_timeout_seconds has correct default value."""
        with MCPClient(mock_transport["transport_callable"]) as client:
            assert client._default_task_poll_timeout_seconds == 300.0

    def test_default_task_poll_timeout_custom_value(self, mock_transport, mock_session):
        """Test that default_task_poll_timeout_seconds can be customized."""
        with MCPClient(mock_transport["transport_callable"], default_task_poll_timeout_seconds=60.0) as client:
            assert client._default_task_poll_timeout_seconds == 60.0


class TestTaskExecutionTimeout:
    """Tests for task execution polling timeout behavior."""

    @pytest.mark.asyncio
    async def test_task_polling_timeout(self, mock_transport, mock_session):
        """Test that task polling times out properly when server takes too long."""
        import asyncio

        # Set up server with task support
        mock_session.get_server_capabilities = MagicMock(return_value=_create_server_capabilities(True))

        # Create a tool with optional task support
        mock_tool = MCPTool(
            name="slow_tool",
            description="A slow tool",
            inputSchema={"type": "object"},
            execution=ToolExecution(taskSupport="optional"),
        )
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[mock_tool], nextCursor=None))

        # Mock task creation
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "test-task-123"
        mock_session.experimental = MagicMock()
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task to never complete (simulate infinite wait)
        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        with MCPClient(
            mock_transport["transport_callable"],
            # Very short timeout for testing
            default_task_poll_timeout_seconds=0.1,
        ) as client:
            # Load tools to cache taskSupport
            client.list_tools_sync()

            result = await client.call_tool_async(
                tool_use_id="test-123", name="slow_tool", arguments={"param": "value"}
            )

            # Should return an error due to timeout
            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_result_retrieval_failure(self, mock_transport, mock_session):
        """Test that get_task_result failures are handled gracefully."""
        # Set up server with task support
        mock_session.get_server_capabilities = MagicMock(return_value=_create_server_capabilities(True))

        # Create a tool with optional task support
        mock_tool = MCPTool(
            name="failing_result_tool",
            description="A tool whose result retrieval fails",
            inputSchema={"type": "object"},
            execution=ToolExecution(taskSupport="optional"),
        )
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[mock_tool], nextCursor=None))

        # Mock task creation
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "test-task-456"
        mock_session.experimental = MagicMock()
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task to complete successfully
        async def successful_poll(task_id):
            yield MagicMock(status="completed", statusMessage=None)

        mock_session.experimental.poll_task = successful_poll

        # Mock get_task_result to raise an exception (simulating race condition/network error)
        mock_session.experimental.get_task_result = AsyncMock(side_effect=Exception("Result expired or network error"))

        with MCPClient(mock_transport["transport_callable"]) as client:
            # Load tools to cache taskSupport
            client.list_tools_sync()

            result = await client.call_tool_async(
                tool_use_id="test-456", name="failing_result_tool", arguments={"param": "value"}
            )

            # Should return an error due to result retrieval failure
            assert result["status"] == "error"
            assert "result retrieval failed" in result["content"][0].get("text", "").lower()

    @pytest.mark.asyncio
    async def test_task_timeout_uses_read_timeout_seconds(self, mock_transport, mock_session):
        """Test that read_timeout_seconds is passed through to task polling timeout."""
        import asyncio
        from datetime import timedelta

        # Set up server with task support
        mock_session.get_server_capabilities = MagicMock(return_value=_create_server_capabilities(True))

        # Create a tool with optional task support
        mock_tool = MCPTool(
            name="timeout_test_tool",
            description="A tool to test timeout passthrough",
            inputSchema={"type": "object"},
            execution=ToolExecution(taskSupport="optional"),
        )
        mock_session.list_tools = AsyncMock(return_value=ListToolsResult(tools=[mock_tool], nextCursor=None))

        # Mock task creation
        mock_create_result = MagicMock()
        mock_create_result.task.taskId = "test-task-timeout"
        mock_session.experimental = MagicMock()
        mock_session.experimental.call_tool_as_task = AsyncMock(return_value=mock_create_result)

        # Mock poll_task to never complete (simulate infinite wait)
        async def infinite_poll(task_id):
            while True:
                await asyncio.sleep(1)
                yield MagicMock(status="running")

        mock_session.experimental.poll_task = infinite_poll

        # Use a 5 minute default, but specify 0.1 second timeout via read_timeout_seconds
        with MCPClient(
            mock_transport["transport_callable"],
            default_task_poll_timeout_seconds=300.0,  # 5 minutes default
        ) as client:
            # Load tools to cache taskSupport
            client.list_tools_sync()

            # Call with explicit timeout - should use this instead of default
            result = await client.call_tool_async(
                tool_use_id="test-timeout",
                name="timeout_test_tool",
                arguments={"param": "value"},
                read_timeout_seconds=timedelta(seconds=0.1),  # 0.1 second timeout
            )

            # Should timeout quickly (0.1 seconds) instead of waiting 5 minutes
            assert result["status"] == "error"
            assert "timed out" in result["content"][0].get("text", "").lower()
