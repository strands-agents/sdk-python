"""Tests for the StrandsA2AExecutor class."""

from unittest.mock import MagicMock

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
async def test_execute_with_text_response(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes text responses correctly."""
    # Setup mock agent response
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "Test response"}]}
    mock_strands_agent.return_value = mock_result

    # Create executor and call execute
    executor = StrandsA2AExecutor(mock_strands_agent)
    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.assert_called_once_with("Test input")

    # Verify event was enqueued
    mock_event_queue.enqueue_event.assert_called_once()
    args, _ = mock_event_queue.enqueue_event.call_args
    event = args[0]
    assert event.parts[0].root.text == "Test response"


@pytest.mark.asyncio
async def test_execute_with_multiple_text_blocks(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute processes multiple text blocks correctly."""
    # Setup mock agent response with multiple text blocks
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": [{"text": "First response"}, {"text": "Second response"}]}
    mock_strands_agent.return_value = mock_result

    # Create executor and call execute
    executor = StrandsA2AExecutor(mock_strands_agent)
    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.assert_called_once_with("Test input")

    # Verify events were enqueued
    assert mock_event_queue.enqueue_event.call_count == 2

    # Check first event
    args1, _ = mock_event_queue.enqueue_event.call_args_list[0]
    event1 = args1[0]
    assert event1.parts[0].root.text == "First response"

    # Check second event
    args2, _ = mock_event_queue.enqueue_event.call_args_list[1]
    event2 = args2[0]
    assert event2.parts[0].root.text == "Second response"


@pytest.mark.asyncio
async def test_execute_with_empty_response(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles empty responses correctly."""
    # Setup mock agent response with empty content
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = {"content": []}
    mock_strands_agent.return_value = mock_result

    # Create executor and call execute
    executor = StrandsA2AExecutor(mock_strands_agent)
    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.assert_called_once_with("Test input")

    # Verify no events were enqueued
    mock_event_queue.enqueue_event.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_no_message(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that execute handles responses with no message correctly."""
    # Setup mock agent response with no message
    mock_result = MagicMock(spec=SAAgentResult)
    mock_result.message = None
    mock_strands_agent.return_value = mock_result

    # Create executor and call execute
    executor = StrandsA2AExecutor(mock_strands_agent)
    await executor.execute(mock_request_context, mock_event_queue)

    # Verify agent was called with correct input
    mock_strands_agent.assert_called_once_with("Test input")

    # Verify no events were enqueued
    mock_event_queue.enqueue_event.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_raises_unsupported_operation_error(mock_strands_agent, mock_request_context, mock_event_queue):
    """Test that cancel raises UnsupportedOperationError."""
    executor = StrandsA2AExecutor(mock_strands_agent)

    with pytest.raises(ServerError) as excinfo:
        await executor.cancel(mock_request_context, mock_event_queue)

    # Verify the error is a ServerError containing an UnsupportedOperationError
    assert isinstance(excinfo.value.error, UnsupportedOperationError)
