import json
import unittest.mock
from typing import cast

import pytest
from pydantic import BaseModel

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


@pytest.fixture
def citations_message():
    """Message with citationsContent block."""
    return {
        "role": "assistant",
        "content": [
            {
                "citationsContent": {
                    "citations": [
                        {
                            "title": "Source Document",
                            "location": {"document": {"pageNumber": 1}},
                            "sourceContent": [{"text": "source text"}],
                        }
                    ],
                    "content": [{"text": "This is cited text from the document."}],
                }
            }
        ],
    }


@pytest.fixture
def mixed_text_and_citations_message():
    """Message with both plain text and citationsContent blocks."""
    return {
        "role": "assistant",
        "content": [
            {"text": "Introduction paragraph"},
            {
                "citationsContent": {
                    "citations": [{"title": "Doc", "location": {}, "sourceContent": []}],
                    "content": [{"text": "Cited content here."}],
                }
            },
            {"text": "Conclusion paragraph"},
        ],
    }


def test__init__(mock_metrics, simple_message: Message):
    """Test that AgentResult can be properly initialized with all required fields."""
    stop_reason: StopReason = "end_turn"
    state = {"key": "value"}

    result = AgentResult(stop_reason=stop_reason, message=simple_message, metrics=mock_metrics, state=state)

    assert result.stop_reason == stop_reason
    assert result.message == simple_message
    assert result.metrics == mock_metrics
    assert result.state == state
    assert result.structured_output is None


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


def test__str__with_citations_content(mock_metrics, citations_message: Message):
    """Test that str() extracts text from citationsContent blocks."""
    result = AgentResult(stop_reason="end_turn", message=citations_message, metrics=mock_metrics, state={})

    message_string = str(result)
    assert message_string == "This is cited text from the document.\n"


def test__str__mixed_text_and_citations_content(mock_metrics, mixed_text_and_citations_message: Message):
    """Test that str() works with both plain text and citationsContent blocks."""
    result = AgentResult(
        stop_reason="end_turn", message=mixed_text_and_citations_message, metrics=mock_metrics, state={}
    )

    message_string = str(result)
    assert message_string == "Introduction paragraph\nCited content here.\nConclusion paragraph\n"


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


# Tests for structured output functionality
class StructuredOutputModel(BaseModel):
    """Test model for structured output."""

    name: str
    value: int
    optional_field: str | None = None


def test__init__with_structured_output(mock_metrics, simple_message: Message):
    """Test that AgentResult can be initialized with structured_output."""
    stop_reason: StopReason = "end_turn"
    state = {"key": "value"}
    structured_output = StructuredOutputModel(name="test", value=42)

    result = AgentResult(
        stop_reason=stop_reason,
        message=simple_message,
        metrics=mock_metrics,
        state=state,
        structured_output=structured_output,
    )

    assert result.stop_reason == stop_reason
    assert result.message == simple_message
    assert result.metrics == mock_metrics
    assert result.state == state
    assert result.structured_output == structured_output
    assert isinstance(result.structured_output, StructuredOutputModel)
    assert result.structured_output.name == "test"
    assert result.structured_output.value == 42


def test__init__structured_output_defaults_to_none(mock_metrics, simple_message: Message):
    """Test that structured_output defaults to None when not provided."""
    result = AgentResult(stop_reason="end_turn", message=simple_message, metrics=mock_metrics, state={})

    assert result.structured_output is None


def test__str__with_structured_output(mock_metrics, simple_message: Message):
    """Test that str() includes BOTH text and structured_output in JSON format."""
    structured_output = StructuredOutputModel(name="test", value=42)

    result = AgentResult(
        stop_reason="end_turn",
        message=simple_message,
        metrics=mock_metrics,
        state={},
        structured_output=structured_output,
    )

    # str() should now include BOTH text AND structured output in JSON format
    message_string = str(result)
    # Output should be valid JSON
    parsed = json.loads(message_string)
    assert parsed["text"] == "Hello world!"
    assert parsed["structured_output"]["name"] == "test"
    assert parsed["structured_output"]["value"] == 42


def test__str__empty_message_with_structured_output(mock_metrics, empty_message: Message):
    """Test that str() returns structured output JSON when message has no text content."""
    structured_output = StructuredOutputModel(name="example", value=123, optional_field="optional")

    result = AgentResult(
        stop_reason="end_turn",
        message=empty_message,
        metrics=mock_metrics,
        state={},
        structured_output=structured_output,
    )

    # When message has no text content, str() should return structured output as JSON
    message_string = str(result)

    # Verify it's the same as the structured output's JSON representation
    assert message_string == structured_output.model_dump_json()

    # Verify it contains the expected data
    assert "example" in message_string
    assert "123" in message_string
    assert "optional" in message_string


def test__str__structured_output_only():
    """Test __str__ with only structured output (no text)."""
    structured = StructuredOutputModel(name="test", value=42)
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": []},
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    # Should return just the structured output JSON
    assert str(result) == '{"name":"test","value":42,"optional_field":null}'


def test__str__both_text_and_structured_output():
    """Test __str__ includes BOTH text and structured output when both exist.

    Output should be JSON-parseable with text and structured_output fields.
    """
    structured = StructuredOutputModel(name="test", value=42)
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "Here is the analysis"}]},
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    output = str(result)
    # Output should be valid JSON
    parsed = json.loads(output)
    assert parsed["text"] == "Here is the analysis"
    assert parsed["structured_output"]["name"] == "test"
    assert parsed["structured_output"]["value"] == 42


def test__str__multiple_text_blocks_with_structured_output():
    """Test __str__ with multiple text blocks and structured output."""
    structured = StructuredOutputModel(name="multi", value=100)
    result = AgentResult(
        stop_reason="end_turn",
        message={
            "role": "assistant",
            "content": [
                {"text": "First paragraph."},
                {"text": "Second paragraph."},
            ],
        },
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    output = str(result)
    # Output should be valid JSON
    parsed = json.loads(output)
    assert "First paragraph." in parsed["text"]
    assert "Second paragraph." in parsed["text"]
    assert parsed["structured_output"]["name"] == "multi"
    assert parsed["structured_output"]["value"] == 100


def test__str__non_text_content_only():
    """Test __str__ with only non-text content (e.g., toolUse)."""
    result = AgentResult(
        stop_reason="tool_use",
        message={
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}],
        },
        metrics=EventLoopMetrics(),
        state={},
    )
    assert str(result) == ""


def test__str__mixed_content_with_structured_output():
    """Test __str__ with mixed content (text + toolUse) and structured output."""
    structured = StructuredOutputModel(name="mixed", value=50)
    result = AgentResult(
        stop_reason="end_turn",
        message={
            "role": "assistant",
            "content": [
                {"text": "Processing complete."},
                {"toolUse": {"toolUseId": "456", "name": "helper", "input": {}}},
            ],
        },
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    output = str(result)
    # Output should be valid JSON
    parsed = json.loads(output)
    assert parsed["text"] == "Processing complete."
    assert parsed["structured_output"]["name"] == "mixed"
    assert parsed["structured_output"]["value"] == 50
    # toolUse should not appear in the text
    assert "toolUse" not in parsed["text"]
    assert "helper" not in parsed["text"]


def test__str__json_parseable():
    """Test that output with both text and structured output is JSON-parseable."""
    structured = StructuredOutputModel(name="parseable", value=99)
    result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "Result text"}]},
        metrics=EventLoopMetrics(),
        state={},
        structured_output=structured,
    )
    output = str(result)
    # Should be valid JSON that can be parsed
    parsed = json.loads(output)
    assert "text" in parsed
    assert "structured_output" in parsed
    assert parsed["text"] == "Result text"
    assert parsed["structured_output"] == {"name": "parseable", "value": 99, "optional_field": None}


def test__str__citations_with_structured_output(mock_metrics, citations_message: Message):
    """Test that str() includes BOTH citationsContent text and structured_output."""
    structured_output = StructuredOutputModel(name="cited", value=77)

    result = AgentResult(
        stop_reason="end_turn",
        message=citations_message,
        metrics=mock_metrics,
        state={},
        structured_output=structured_output,
    )

    message_string = str(result)
    # Output should be valid JSON with both text and structured output
    parsed = json.loads(message_string)
    assert parsed["text"] == "This is cited text from the document."
    assert parsed["structured_output"]["name"] == "cited"
    assert parsed["structured_output"]["value"] == 77


def test__str__mixed_text_citations_with_structured_output(mock_metrics, mixed_text_and_citations_message: Message):
    """Test that str() handles plain text, citations, and structured output together."""
    structured_output = StructuredOutputModel(name="complex", value=999)

    result = AgentResult(
        stop_reason="end_turn",
        message=mixed_text_and_citations_message,
        metrics=mock_metrics,
        state={},
        structured_output=structured_output,
    )

    message_string = str(result)
    # Output should be valid JSON
    parsed = json.loads(message_string)
    assert parsed["text"] == "Introduction paragraph\nCited content here.\nConclusion paragraph"
    assert parsed["structured_output"]["name"] == "complex"
    assert parsed["structured_output"]["value"] == 999
