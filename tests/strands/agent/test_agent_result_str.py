"""Tests for AgentResult.__str__ method with Option 1 behavior.

This module tests that __str__ properly includes both text and structured output
when both exist (fix for issue #1461).
"""

import pytest
from pydantic import BaseModel

from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics


class SampleOutput(BaseModel):
    """Sample structured output model for testing."""

    name: str
    value: int


class TestAgentResultStrOption1:
    """Tests for Option 1 behavior: __str__ includes both text and structured output."""

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
        
        This is the key fix for issue #1461 - Option 1.
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
        # Should include both text AND structured output
        assert "Here is the analysis" in output
        assert "[Structured Output]" in output
        assert '{"name":"test","value":42}' in output

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
        assert "First paragraph." in output
        assert "Second paragraph." in output
        assert "[Structured Output]" in output
        assert '{"name":"multi","value":100}' in output

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
        assert "Processing complete." in output
        assert "[Structured Output]" in output
        assert '{"name":"mixed","value":50}' in output
        # toolUse should not appear in string output
        assert "toolUse" not in output
        assert "helper" not in output

    def test_str_format_structure(self):
        """Test the exact format of __str__ output with both text and structured output."""
        structured = SampleOutput(name="format", value=99)
        result = AgentResult(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"text": "Result text"}]},
            metrics=EventLoopMetrics(),
            state={},
            structured_output=structured,
        )
        output = str(result)
        # Verify the format: text followed by structured output section
        lines = output.strip().split("\n")
        assert lines[0] == "Result text"
        assert "[Structured Output]" in output
        assert lines[-1] == '{"name":"format","value":99}'
