"""Tests for AgentResult.to_graph_input_text() method."""

import unittest.mock

import pytest
from pydantic import BaseModel

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.content import Message


@pytest.fixture
def mock_metrics():
    return unittest.mock.Mock(spec=EventLoopMetrics)


class TestModel(BaseModel):
    """Test Pydantic model for structured output."""

    name: str
    value: int
    description: str = "test"


def test_to_graph_input_text_with_text_only(mock_metrics):
    """Test that to_graph_input_text returns only text when no structured output exists."""
    message: Message = {"role": "assistant", "content": [{"text": "Hello from agent"}]}
    result = AgentResult(stop_reason="end_turn", message=message, metrics=mock_metrics, state={})

    output = result.to_graph_input_text()

    assert output == "Hello from agent\n"
    assert "Structured Output:" not in output


def test_to_graph_input_text_with_structured_output_only(mock_metrics):
    """Test that to_graph_input_text returns structured output when no text exists."""
    message: Message = {"role": "assistant", "content": []}
    structured_output = TestModel(name="test_entity", value=42, description="A test object")
    result = AgentResult(
        stop_reason="end_turn", message=message, metrics=mock_metrics, state={}, structured_output=structured_output
    )

    output = result.to_graph_input_text()

    # Should return the JSON representation of structured output
    assert "test_entity" in output
    assert "42" in output
    assert "A test object" in output
    # Should NOT have "Structured Output:" prefix when there's no text
    assert not output.startswith("\nStructured Output:")


def test_to_graph_input_text_with_both_text_and_structured_output(mock_metrics):
    """Test that to_graph_input_text returns both text and structured output when both exist."""
    message: Message = {"role": "assistant", "content": [{"text": "Analysis complete"}]}
    structured_output = TestModel(name="result", value=100, description="Final result")
    result = AgentResult(
        stop_reason="end_turn", message=message, metrics=mock_metrics, state={}, structured_output=structured_output
    )

    output = result.to_graph_input_text()

    # Should contain the text
    assert "Analysis complete\n" in output
    # Should contain structured output section
    assert "\nStructured Output:\n" in output
    # Should contain structured data
    assert "result" in output
    assert "100" in output
    assert "Final result" in output


def test_to_graph_input_text_with_multiple_text_blocks(mock_metrics):
    """Test that to_graph_input_text handles multiple text blocks correctly."""
    message: Message = {
        "role": "assistant",
        "content": [{"text": "First part"}, {"text": "Second part"}, {"text": "Third part"}],
    }
    result = AgentResult(stop_reason="end_turn", message=message, metrics=mock_metrics, state={})

    output = result.to_graph_input_text()

    assert "First part\n" in output
    assert "Second part\n" in output
    assert "Third part\n" in output


def test_to_graph_input_text_empty_message(mock_metrics):
    """Test that to_graph_input_text handles empty messages."""
    message: Message = {"role": "assistant", "content": []}
    result = AgentResult(stop_reason="end_turn", message=message, metrics=mock_metrics, state={})

    output = result.to_graph_input_text()

    assert output == ""


def test_to_graph_input_text_ignores_non_text_content(mock_metrics):
    """Test that to_graph_input_text ignores non-text content blocks."""
    message: Message = {
        "role": "assistant",
        "content": [{"text": "Valid text"}, {"toolUse": {"name": "tool"}}, {"text": "More text"}],
    }
    result = AgentResult(stop_reason="end_turn", message=message, metrics=mock_metrics, state={})

    output = result.to_graph_input_text()

    assert "Valid text\n" in output
    assert "More text\n" in output
    assert "toolUse" not in output


def test_to_graph_input_text_with_complex_structured_output(mock_metrics):
    """Test to_graph_input_text with nested structured output."""

    class NestedModel(BaseModel):
        """Nested Pydantic model."""

        items: list[str]
        count: int
        metadata: dict[str, str]

    message: Message = {"role": "assistant", "content": [{"text": "Processing complete"}]}
    structured_output = NestedModel(items=["a", "b", "c"], count=3, metadata={"key": "value"})
    result = AgentResult(
        stop_reason="end_turn", message=message, metrics=mock_metrics, state={}, structured_output=structured_output
    )

    output = result.to_graph_input_text()

    # Should contain text
    assert "Processing complete\n" in output
    # Should contain structured output header
    assert "\nStructured Output:\n" in output
    # Should contain nested data
    assert '"items":' in output or "'items':" in output
    assert '"count":3' in output or "'count':3" in output


def test_str_method_unchanged(mock_metrics):
    """Verify that __str__ method behavior is NOT affected by our changes."""
    message: Message = {"role": "assistant", "content": [{"text": "Hello"}]}
    structured_output = TestModel(name="test", value=42)
    result = AgentResult(
        stop_reason="end_turn", message=message, metrics=mock_metrics, state={}, structured_output=structured_output
    )

    str_output = str(result)

    # __str__ should only return text, ignoring structured_output when text exists
    assert str_output == "Hello\n"
    assert "Structured Output:" not in str_output
    assert "test" not in str_output


def test_str_vs_to_graph_input_text_difference(mock_metrics):
    """Test the key difference between __str__ and to_graph_input_text."""
    message: Message = {"role": "assistant", "content": [{"text": "Text content"}]}
    structured_output = TestModel(name="data", value=99)
    result = AgentResult(
        stop_reason="end_turn", message=message, metrics=mock_metrics, state={}, structured_output=structured_output
    )

    str_output = str(result)
    graph_output = result.to_graph_input_text()

    # __str__ returns only text (original behavior preserved)
    assert str_output == "Text content\n"
    assert "data" not in str_output

    # to_graph_input_text returns both text and structured output
    assert "Text content\n" in graph_output
    assert "Structured Output:" in graph_output
    assert "data" in graph_output
    assert "99" in graph_output
