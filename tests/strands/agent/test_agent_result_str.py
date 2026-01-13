"""Tests for AgentResult.__str__ method with Option 1 behavior.

This module tests that __str__ properly includes both text and structured output
when both exist (fix for issue #1461).

The output format when both text and structured output exist is JSON:
{"text": "...", "structured_output": {...}}

This allows users to parse the output programmatically.
"""

import json

from pydantic import BaseModel

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics


class SampleOutput(BaseModel):
    """Sample structured output model for testing."""

    name: str
    value: int


class TestAgentResultStrOption1:
    """Tests for Option 1 behavior: __str__ includes both text and structured output in JSON format."""

    def test_str_text_only(self):
        """Test __str__ with only text content."""
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "Hello world"}]},
            metrics=EventLoopMetrics(),
            state={},
        )
        assert str(result) == "Hello world\n"

    def test_str_structured_output_only(self):
        """Test __str__ with only structured output (no text)."""
        structured = SampleOutput(name="test", value=42)
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": []},
            metrics=EventLoopMetrics(),
            state={},
            structured_output=structured,
        )
        assert str(result) == '{"name":"test","value":42}'

    def test_str_both_text_and_structured_output(self):
        """Test __str__ includes BOTH text and structured output when both exist.

        This is the key fix for issue #1461 - Option 1 with JSON format.
        Output should be JSON-parseable.
        """
        structured = SampleOutput(name="test", value=42)
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

    def test_str_multiple_text_blocks_with_structured_output(self):
        """Test __str__ with multiple text blocks and structured output."""
        structured = SampleOutput(name="multi", value=100)
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

    def test_str_empty_message_no_structured_output(self):
        """Test __str__ with empty message and no structured output."""
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": []},
            metrics=EventLoopMetrics(),
            state={},
        )
        assert str(result) == ""

    def test_str_non_text_content_only(self):
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

    def test_str_mixed_content_with_structured_output(self):
        """Test __str__ with mixed content (text + toolUse) and structured output."""
        structured = SampleOutput(name="mixed", value=50)
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

    def test_str_json_parseable(self):
        """Test that output with both text and structured output is JSON-parseable."""
        structured = SampleOutput(name="parseable", value=99)
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
        assert parsed["structured_output"] == {"name": "parseable", "value": 99}
