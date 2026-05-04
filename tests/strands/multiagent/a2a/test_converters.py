"""Tests for A2A converter functions."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from a2a.types import Message as A2AMessage
from a2a.types import Part, Role, TaskArtifactUpdateEvent, TaskStatusUpdateEvent, TextPart

from strands.agent.agent_result import AgentResult
from strands.multiagent.a2a._converters import (
    convert_content_blocks_to_parts,
    convert_input_to_message,
    convert_response_to_agent_result,
)


def test_convert_string_input():
    """Test converting string input to A2A message."""
    message = convert_input_to_message("Hello")

    assert isinstance(message, A2AMessage)
    assert message.role == Role.user
    assert len(message.parts) == 1
    assert message.parts[0].root.text == "Hello"


def test_convert_message_list_input():
    """Test converting message list input to A2A message."""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    message = convert_input_to_message(messages)

    assert isinstance(message, A2AMessage)
    assert message.role == Role.user
    assert len(message.parts) == 1


def test_convert_content_blocks_input():
    """Test converting content blocks input to A2A message."""
    content_blocks = [{"text": "Hello"}, {"text": "World"}]

    message = convert_input_to_message(content_blocks)

    assert isinstance(message, A2AMessage)
    assert len(message.parts) == 2


def test_convert_unsupported_input():
    """Test that unsupported input types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported input type"):
        convert_input_to_message(123)


def test_convert_interrupt_response_raises_error():
    """Test that InterruptResponseContent raises explicit error."""
    interrupt_responses = [{"interruptResponse": {"interruptId": "123", "response": "A"}}]

    with pytest.raises(ValueError, match="InterruptResponseContent is not supported for A2AAgent"):
        convert_input_to_message(interrupt_responses)


def test_convert_content_blocks_to_parts():
    """Test converting content blocks to A2A parts."""
    content_blocks = [{"text": "Hello"}, {"text": "World"}]

    parts = convert_content_blocks_to_parts(content_blocks)

    assert len(parts) == 2
    assert parts[0].root.text == "Hello"
    assert parts[1].root.text == "World"


def test_convert_a2a_message_response():
    """Test converting A2A message response to AgentResult."""
    a2a_message = A2AMessage(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[Part(TextPart(kind="text", text="Response"))],
    )

    result = convert_response_to_agent_result(a2a_message)

    assert isinstance(result, AgentResult)
    assert result.message["role"] == "assistant"
    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Response"


def test_convert_task_response():
    """Test converting task response to AgentResult."""
    mock_task = MagicMock()
    mock_artifact = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Task response"
    mock_artifact.parts = [mock_part]
    mock_task.artifacts = [mock_artifact]

    result = convert_response_to_agent_result((mock_task, None))

    assert isinstance(result, AgentResult)
    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Task response"


def test_convert_multiple_parts_response():
    """Test converting response with multiple parts to separate content blocks."""
    a2a_message = A2AMessage(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[
            Part(TextPart(kind="text", text="First")),
            Part(TextPart(kind="text", text="Second")),
        ],
    )

    result = convert_response_to_agent_result(a2a_message)

    assert len(result.message["content"]) == 2
    assert result.message["content"][0]["text"] == "First"
    assert result.message["content"][1]["text"] == "Second"


# --- New tests for coverage ---


def test_convert_message_list_finds_last_user_message():
    """Test that message list conversion finds the last user message."""
    messages = [
        {"role": "user", "content": [{"text": "First"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
        {"role": "user", "content": [{"text": "Second"}]},
    ]

    message = convert_input_to_message(messages)

    assert message.parts[0].root.text == "Second"


def test_convert_content_blocks_skips_non_text():
    """Test that non-text content blocks are skipped."""
    content_blocks = [{"text": "Hello"}, {"image": "data"}, {"text": "World"}]

    parts = convert_content_blocks_to_parts(content_blocks)

    assert len(parts) == 2


def test_convert_task_artifact_update_event():
    """Test converting TaskArtifactUpdateEvent to AgentResult."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Streamed artifact"
    mock_artifact = MagicMock()
    mock_artifact.parts = [mock_part]

    mock_event = MagicMock(spec=TaskArtifactUpdateEvent)
    mock_event.artifact = mock_artifact

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert result.message["content"][0]["text"] == "Streamed artifact"


def test_convert_task_status_update_event():
    """Test converting TaskStatusUpdateEvent to AgentResult."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Status message"
    mock_message = MagicMock()
    mock_message.parts = [mock_part]
    mock_status = MagicMock()
    mock_status.message = mock_message

    mock_event = MagicMock(spec=TaskStatusUpdateEvent)
    mock_event.status = mock_status

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert result.message["content"][0]["text"] == "Status message"


def test_convert_task_status_update_event_no_message_falls_back_to_task_artifacts():
    """Test that TaskStatusUpdateEvent with no message falls back to task.artifacts."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Artifact content"
    mock_artifact = MagicMock()
    mock_artifact.parts = [mock_part]
    mock_task.artifacts = [mock_artifact]

    mock_event = MagicMock(spec=TaskStatusUpdateEvent)
    mock_status = MagicMock()
    mock_status.message = None
    mock_event.status = mock_status

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Artifact content"


def test_convert_task_artifact_update_event_empty_parts_falls_back_to_task_artifacts():
    """Test that TaskArtifactUpdateEvent with empty parts falls back to task.artifacts."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Full artifact content"
    mock_artifact = MagicMock()
    mock_artifact.parts = [mock_part]
    mock_task.artifacts = [mock_artifact]

    mock_event = MagicMock(spec=TaskArtifactUpdateEvent)
    mock_event_artifact = MagicMock()
    mock_event_artifact.parts = []
    mock_event.artifact = mock_event_artifact

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Full artifact content"


def test_convert_response_handles_missing_data():
    """Test that response conversion handles missing/malformed data gracefully."""
    # TaskArtifactUpdateEvent with no artifact
    mock_event = MagicMock(spec=TaskArtifactUpdateEvent)
    mock_event.artifact = None
    result = convert_response_to_agent_result((MagicMock(), mock_event))
    assert len(result.message["content"]) == 0

    # TaskStatusUpdateEvent with no status
    mock_event = MagicMock(spec=TaskStatusUpdateEvent)
    mock_event.status = None
    result = convert_response_to_agent_result((MagicMock(), mock_event))
    assert len(result.message["content"]) == 0

    # Task artifact without parts attribute
    mock_task = MagicMock()
    mock_artifact = MagicMock(spec=[])
    del mock_artifact.parts
    mock_task.artifacts = [mock_artifact]
    result = convert_response_to_agent_result((mock_task, None))
    assert len(result.message["content"]) == 0


# =========================================================================
# NEW TESTS: Lifecycle State Mapping
# =========================================================================


def test_convert_response_completed_state_maps_to_end_turn():
    """Test that completed state maps to end_turn stop_reason."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Part, TextPart

    task = MagicMock()
    task.artifacts = None

    status = TaskStatus(state=TaskState.completed, message=None)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "end_turn"


def test_convert_response_failed_state_maps_to_end_turn():
    """Test that failed state maps to end_turn stop_reason with error content."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Message, Part, TextPart, Role

    task = MagicMock()
    task.artifacts = None

    # Create a status message with error info
    error_part = MagicMock()
    error_part.root = MagicMock()
    error_part.root.text = "Agent execution failed: timeout"

    error_message = MagicMock(spec=Message)
    error_message.parts = [error_part]

    status = TaskStatus(state=TaskState.failed, message=error_message)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "end_turn"
    assert result.state.get("a2a_task_state") == "failed"
    assert "Agent execution failed" in result.message["content"][0]["text"]


def test_convert_response_input_required_maps_to_interrupt():
    """Test that input_required state maps to interrupt stop_reason."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Message, Part, TextPart, Role

    task = MagicMock()
    task.artifacts = None

    input_part = MagicMock()
    input_part.root = MagicMock()
    input_part.root.text = "Agent requires input:\n- approval: Need confirmation"

    input_message = MagicMock(spec=Message)
    input_message.parts = [input_part]

    status = TaskStatus(state=TaskState.input_required, message=input_message)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "interrupt"
    assert result.state.get("a2a_task_state") == "input-required"
    assert "approval" in result.message["content"][0]["text"]


def test_convert_response_canceled_state_maps_to_end_turn():
    """Test that canceled state maps to end_turn stop_reason."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus

    task = MagicMock()
    task.artifacts = None

    status = TaskStatus(state=TaskState.canceled, message=None)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "end_turn"
    assert result.state.get("a2a_task_state") == "canceled"


def test_convert_response_rejected_state_maps_to_end_turn():
    """Test that rejected state maps to end_turn stop_reason."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus

    task = MagicMock()
    task.artifacts = None

    status = TaskStatus(state=TaskState.rejected, message=None)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "end_turn"
    assert result.state.get("a2a_task_state") == "rejected"


def test_convert_response_auth_required_maps_to_interrupt():
    """Test that auth_required state maps to interrupt stop_reason."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus

    task = MagicMock()
    task.artifacts = None

    status = TaskStatus(state=TaskState.auth_required, message=None)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    result = convert_response_to_agent_result((task, update_event))
    assert result.stop_reason == "interrupt"
    assert result.state.get("a2a_task_state") == "auth-required"


def test_extract_task_state_from_status_update():
    """Test _extract_task_state helper."""
    from unittest.mock import MagicMock
    from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus
    from strands.multiagent.a2a._converters import _extract_task_state

    task = MagicMock()
    status = TaskStatus(state=TaskState.failed, message=None)
    update_event = MagicMock(spec=TaskStatusUpdateEvent)
    update_event.status = status

    state = _extract_task_state((task, update_event))
    assert state == TaskState.failed


def test_extract_task_state_from_message_returns_none():
    """Test _extract_task_state returns None for Message responses."""
    from unittest.mock import MagicMock
    from a2a.types import Message
    from strands.multiagent.a2a._converters import _extract_task_state

    message = MagicMock(spec=Message)
    state = _extract_task_state(message)
    assert state is None
