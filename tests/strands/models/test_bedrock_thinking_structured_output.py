"""Tests for Bedrock thinking mode with structured output."""

import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel

from strands.models.bedrock import BedrockModel


class TestModel(BaseModel):
    """Test model for structured output."""
    name: str
    value: int


@pytest.fixture
def bedrock_model_with_thinking():
    """Create a BedrockModel with thinking enabled."""
    return BedrockModel(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        additional_request_fields={
            "thinking": {
                "type": "enabled",
                "budget_tokens": 5000
            }
        }
    )


def test_format_request_removes_thinking_when_forcing_tool(bedrock_model_with_thinking):
    """Test that thinking is removed from request when tool_choice forces tool use."""
    messages = [{"role": "user", "content": [{"text": "test"}]}]
    tool_specs = [
        {
            "name": "TestTool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    # When forcing tool use (not auto), thinking should be removed
    tool_choice = {"any": {}}
    
    request = bedrock_model_with_thinking._format_request(
        messages=messages,
        tool_specs=tool_specs,
        system_prompt_content=[],
        tool_choice=tool_choice
    )
    
    # Verify thinking is NOT in the request
    assert "additionalModelRequestFields" in request
    additional_fields = request["additionalModelRequestFields"]
    assert "thinking" not in additional_fields


def test_format_request_keeps_thinking_with_auto_tool_choice(bedrock_model_with_thinking):
    """Test that thinking is preserved when tool_choice is auto."""
    messages = [{"role": "user", "content": [{"text": "test"}]}]
    tool_specs = [
        {
            "name": "TestTool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    # With auto tool choice, thinking should be preserved
    tool_choice = {"auto": {}}
    
    request = bedrock_model_with_thinking._format_request(
        messages=messages,
        tool_specs=tool_specs,
        system_prompt_content=[],
        tool_choice=tool_choice
    )
    
    # Verify thinking IS in the request
    assert "additionalModelRequestFields" in request
    additional_fields = request["additionalModelRequestFields"]
    assert "thinking" in additional_fields
    assert additional_fields["thinking"]["type"] == "enabled"
    assert additional_fields["thinking"]["budget_tokens"] == 5000


def test_format_request_keeps_thinking_with_no_tool_choice(bedrock_model_with_thinking):
    """Test that thinking is preserved when tool_choice is None."""
    messages = [{"role": "user", "content": [{"text": "test"}]}]
    tool_specs = [
        {
            "name": "TestTool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    # With no tool choice (None), thinking should be preserved
    request = bedrock_model_with_thinking._format_request(
        messages=messages,
        tool_specs=tool_specs,
        system_prompt_content=[],
        tool_choice=None
    )
    
    # Verify thinking IS in the request
    assert "additionalModelRequestFields" in request
    additional_fields = request["additionalModelRequestFields"]
    assert "thinking" in additional_fields


def test_format_request_preserves_other_additional_fields_when_removing_thinking(bedrock_model_with_thinking):
    """Test that other additionalRequestFields are preserved when thinking is removed."""
    # Add another field to additional_request_fields
    bedrock_model_with_thinking.config["additional_request_fields"]["custom_field"] = "custom_value"
    
    messages = [{"role": "user", "content": [{"text": "test"}]}]
    tool_specs = [
        {
            "name": "TestTool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    # Force tool use
    tool_choice = {"any": {}}
    
    request = bedrock_model_with_thinking._format_request(
        messages=messages,
        tool_specs=tool_specs,
        system_prompt_content=[],
        tool_choice=tool_choice
    )
    
    # Verify thinking is removed but other fields are preserved
    additional_fields = request["additionalModelRequestFields"]
    assert "thinking" not in additional_fields
    assert "custom_field" in additional_fields
    assert additional_fields["custom_field"] == "custom_value"


def test_format_request_without_thinking_config():
    """Test that models without thinking config work normally with forced tool choice."""
    model = BedrockModel(model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0")
    
    messages = [{"role": "user", "content": [{"text": "test"}]}]
    tool_specs = [
        {
            "name": "TestTool",
            "description": "A test tool",
            "inputSchema": {"type": "object", "properties": {}}
        }
    ]
    
    # Force tool use
    tool_choice = {"any": {}}
    
    request = model._format_request(
        messages=messages,
        tool_specs=tool_specs,
        system_prompt_content=[],
        tool_choice=tool_choice
    )
    
    # Should work fine (no thinking to remove)
    assert "toolConfig" in request
    assert request["toolConfig"]["toolChoice"] == {"any": {}}
