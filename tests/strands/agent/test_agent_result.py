import unittest.mock
from typing import cast

import pytest

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import Message
from strands.types.streaming import StopReason


@pytest.fixture
def mock_metrics():
    return unittest.mock.Mock(spec=EventLoopMetrics)


@pytest.fixture
def simple_message():
    return {"role": "assistant", "content": [{"text": "Hello world!"}]}


@pytest.fixture
def complex_message():
    return {
        "role": "assistant",
        "content": [
            {"text": "First paragraph"},
            {"text": "Second paragraph"},
            {"non_text_content": "This should be ignored"},
            {"text": "Third paragraph"},
        ],
    }


@pytest.fixture
def empty_message():
    return {"role": "assistant", "content": []}


def test__init__(mock_metrics, simple_message: Message):
    """Test that AgentResult can be properly initialized with all required fields."""
    stop_reason: StopReason = "end_turn"
    state = {"key": "value"}

    result = AgentResult(stop_reason=stop_reason, message=simple_message, metrics=mock_metrics, state=state)

    assert result.stop_reason == stop_reason
    assert result.message == simple_message
    assert result.metrics == mock_metrics
    assert result.state == state


def test__str__simple(mock_metrics, simple_message: Message):
    """Test that str() works with a simple message."""
    result = AgentResult(stop_reason="end_turn", message=simple_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "Hello world!\n"


def test__str__complex(mock_metrics, complex_message: Message):
    """Test that str() works with a complex message with multiple text blocks."""
    result = AgentResult(stop_reason="end_turn", message=complex_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "First paragraph\nSecond paragraph\nThird paragraph\n"


def test__str__empty(mock_metrics, empty_message: Message):
    """Test that str() works with an empty message."""
    result = AgentResult(stop_reason="end_turn", message=empty_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == ""


def test__str__no_content(mock_metrics):
    """Test that str() works with a message that has no content field."""
    message_without_content = cast(Message, {"role": "assistant"})

    result = AgentResult(stop_reason="end_turn", message=message_without_content, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == ""


def test__str__non_dict_content(mock_metrics):
    """Test that str() handles non-dictionary content items gracefully."""
    message_with_non_dict = cast(
        Message,
        {"role": "assistant", "content": [{"text": "Valid text"}, "Not a dictionary", {"text": "More valid text"}]},
    )

    result = AgentResult(stop_reason="end_turn", message=message_with_non_dict, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "Valid text\nMore valid text\n"


def test_to_dict(mock_metrics, simple_message: Message):
    """Test that to_dict serializes AgentResult correctly."""
    result = AgentResult(stop_reason="end_turn", message=simple_message, metrics=mock_metrics, state={"key": "value"})

    data = result.to_dict()

    assert data == {
        "type": "agent_result",
        "message": simple_message,
        "stop_reason": "end_turn",
    }


def test_from_dict():
    """Test that from_dict works with valid data."""
    data = {
        "type": "agent_result",
        "message": {"role": "assistant", "content": [{"text": "Test response"}]},
        "stop_reason": "end_turn",
    }

    result = AgentResult.from_dict(data)

    assert result.message == data["message"]
    assert result.stop_reason == data["stop_reason"]
    assert isinstance(result.metrics, EventLoopMetrics)
    assert result.state == {}


def test_roundtrip_serialization(mock_metrics, complex_message: Message):
    """Test that to_dict() and from_dict() work together correctly."""
    original = AgentResult(
        stop_reason="max_tokens", message=complex_message, metrics=mock_metrics, state={"test": "data"}
    )

    # Serialize and deserialize
    data = original.to_dict()
    restored = AgentResult.from_dict(data)

    assert restored.message == original.message
    assert restored.stop_reason == original.stop_reason
    assert isinstance(restored.metrics, EventLoopMetrics)
    assert restored.state == {}  # State is not serialized
