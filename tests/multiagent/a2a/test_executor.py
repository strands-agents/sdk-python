"""Tests for the StrandsA2AExecutor class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import AgentCapabilities, UnsupportedOperationError
from a2a.utils.errors import ServerError

from strands.agent.agent_result import AgentResult as SAAgentResult
from strands.multiagent.a2a.executor import StrandsA2AExecutor


def test_executor_initialization(mock_strands_agent):
    """Test that StrandsA2AExecutor initializes correctly."""
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    assert executor.agent == mock_strands_agent
    assert executor.capabilities == capabilities


def test_executor_initialization_with_streaming(mock_strands_agent):
    """Test that StrandsA2AExecutor initializes correctly with streaming enabled."""
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    assert executor.agent == mock_strands_agent
    assert executor.capabilities.streaming is True


@pytest.mark.asyncio
async def test_execute_sync_mode_with_text_response(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes text responses correctly in sync mode."""
    # Setup mock agent response
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "Test response"}]}
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify event was enqueued for task completion
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_sync_mode_with_multiple_text_blocks(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes multiple text blocks correctly in sync mode."""
    # Setup mock agent response with multiple text blocks
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "First response"}, {"text": "Second response"}]}
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_sync_mode_with_string_content(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles string content correctly in sync mode."""
    # Setup mock agent response with string content
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": "Simple string response"}
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify events were enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_sync_mode_with_empty_response(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles empty responses correctly in sync mode."""
    # Setup mock agent response with empty content
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": []}
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify completion event was still enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_sync_mode_with_no_message(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles responses with no message correctly in sync mode."""
    # Setup mock agent response with no message
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = None
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify completion event was still enqueued
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
async def test_execute_streaming_mode_with_data_events(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes streaming data events correctly."""

    # Setup mock streaming response
    async def mock_stream():
        yield {"data": "Streaming chunk 1"}
        yield {"data": "Streaming chunk 2"}
        yield {"result": None}

    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream())

    # Create executor with streaming capabilities
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

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
    """Test that execute processes streaming result events correctly."""
    # Setup mock streaming response with result
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "Final result"}]}

    async def mock_stream():
        yield {"data": "Streaming chunk"}
        yield {"result": mock_result}

    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream())

    # Create executor with streaming capabilities
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

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
    """Test that execute handles empty streaming data correctly."""

    # Setup mock streaming response with empty data
    async def mock_stream():
        yield {"data": ""}
        yield {"data": None}
        yield {"result": None}

    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream())

    # Create executor with streaming capabilities
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

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
    """Test that execute handles unexpected streaming events correctly."""

    # Setup mock streaming response with unexpected event
    async def mock_stream():
        yield {"data": "Valid chunk"}
        yield {"unexpected": "event"}
        yield {"result": None}

    mock_strands_agent.stream_async = MagicMock(return_value=mock_stream())

    # Create executor with streaming capabilities
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.stream_async.assert_called_once_with("Test input")

    # Should still complete successfully despite unexpected event
    mock_event_queue.enqueue_event.assert_called()


@pytest.mark.asyncio
@patch("strands.multiagent.a2a.executor.new_task")
async def test_execute_creates_task_when_none_exists(
    mock_new_task, mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that execute creates a task when none exists in the context."""
    # Setup mock agent response
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "Test response"}]}
    mock_strands_agent.invoke_async = AsyncMock(return_value=mock_result)

    # Setup mock task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_new_task.return_value = mock_task

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # No current task in context
    mock_request_context.current_task = None

    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")

    # Verify task creation and completion events were enqueued
    assert mock_event_queue.enqueue_event.call_count >= 1
    mock_new_task.assert_called_once()


@pytest.mark.asyncio
async def test_execute_sync_mode_handles_agent_exception(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles agent exceptions correctly in sync mode."""
    # Setup mock agent to raise exception
    mock_strands_agent.invoke_async = AsyncMock(side_effect=Exception("Agent error"))

    # Create executor with sync capabilities
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    with pytest.raises(ServerError):
        await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called
    mock_strands_agent.invoke_async.assert_called_once_with("Test input")


@pytest.mark.asyncio
async def test_execute_streaming_mode_handles_agent_exception(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that execute handles agent exceptions correctly in streaming mode."""
    # Setup mock agent to raise exception
    mock_strands_agent.stream_async = MagicMock(side_effect=Exception("Agent error"))

    # Create executor with streaming capabilities
    capabilities = AgentCapabilities(streaming=True)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

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
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)


@pytest.mark.asyncio
async def test_handle_agent_result_with_none_result(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that _handle_agent_result handles None result correctly."""
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()

    # Call _handle_agent_result with None
    await executor._handle_agent_result(None, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()


@pytest.mark.asyncio
async def test_handle_agent_result_with_result_but_no_message(
    mock_strands_agent, mock_request_context, mock_event_queue
):
    """Test that _handle_agent_result handles result with no message correctly."""
    capabilities = AgentCapabilities(streaming=False)
    executor = StrandsA2AExecutor(mock_strands_agent, capabilities)

    # Mock the task creation
    mock_task = MagicMock()
    mock_task.id = "test-task-id"
    mock_task.contextId = "test-context-id"
    mock_request_context.current_task = mock_task

    # Mock TaskUpdater
    mock_updater = MagicMock()
    mock_updater.complete = AsyncMock()

    # Create result with no message
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = None

    # Call _handle_agent_result
    await executor._handle_agent_result(mock_result, mock_updater)

    # Verify completion was called
    mock_updater.complete.assert_called_once()
