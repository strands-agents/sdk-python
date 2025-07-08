"""Tests for the StrandsA2AExecutor class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import UnsupportedOperationError
from a2a.utils.errors import ServerError

from strands.agent.agent_result import AgentResult as SAAgentResult
from strands.multiagent.a2a.executor import StrandsA2AExecutor


def test_executor_initialization(mock_strands_agent):
    """Test that StrandsA2AExecutor initializes correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    assert executor.agent == mock_strands_agent


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_data_events(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes data events correctly in streaming mode."""

    async def mock_stream(user_input):
        """Mock streaming function that yields data events."""
        yield {"data": "First chunk"}
        yield {"data": "Second chunk"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream("Test input"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.stream_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_result_event(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes result events correctly in streaming mode."""

    async def mock_stream(user_input):
        """Mock streaming function that yields only result event."""
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream("Test input"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.stream_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_empty_data(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles empty data events correctly in streaming mode."""

    async def mock_stream(user_input):
        """Mock streaming function that yields empty data."""
        yield {"data": ""}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream("Test input"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.stream_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_unexpected_event(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles unexpected events correctly in streaming mode."""

    async def mock_stream(user_input):
        """Mock streaming function that yields unexpected event."""
        yield {"unexpected": "event"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream("Test input"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.stream_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_creates_task_when_none_exists(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute creates a new task when none exists."""

    async def mock_stream(user_input):
        """Mock streaming function that yields data events."""
        yield {"data": "Test chunk"}
        yield {"result": MagicMock(spec=SAAgentResult)}

    # Setup mock agent streaming
    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream("Test input"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock no existing task
    mock_request_context.current_task = None

    with patch("strands.multiagent.a2a.executor.new_task") as mock_new_task:
        mock_new_task.return_value = MagicMock(id="new-task-id", contextId="new-context-id")

        await executor.execute(mock_request_context, mock_event_queue)

    # Verify task creation and completion events were enqueued
    assert mock_event_queue.enqueue_event.call_count >= 1
    mock_new_task.assert_called_once()


@pytest.mark.asyncio
async def test_execute_streaming_mode_handles_agent_exception(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that execute handles agent exceptions correctly in streaming mode."""

    # Setup mock agent to raise exception when stream_async is called
    mock_strands_agent.stream_async = MagicMock(side_effect=Exception("Agent error"))

    # Create executor
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    with pytest.raises(ServerError):
        await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called
    mock_strands_agent.stream_async.assert_called_once_with("Test input")


@pytest.mark.asyncio
async def test_cancel_raises_unsupported_operation_error(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that cancel raises UnsupportedOperationError."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_handle_agent_result_with_none_result(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that _handle_agent_result handles None result correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()
    mock_updater.add_artifact = AsyncMock()

    # Call _handle_agent_result with None
    await executor._handle_agent_result(None, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()


@pytest.mark.asyncio
async def test_handle_agent_result_with_result_but_no_message(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that _handle_agent_result handles result with no message correctly."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()
    mock_updater.add_artifact = AsyncMock()

    # Create result with no message
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = None

    # Call _handle_agent_result
    await executor._handle_agent_result(mock_result, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()
