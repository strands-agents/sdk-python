"""Tests for strands.types.tools module."""

import pytest

from strands.types.tools import ToolUse


def test_tool_use_required_fields():
    """Test that ToolUse can be created with only required fields."""
    tool_use: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {"key": "value"},
    }

    assert tool_use["toolUseId"] == "test-id"
    assert tool_use["name"] == "test_tool"
    assert tool_use["input"] == {"key": "value"}
    assert "thoughtSignature" not in tool_use


def test_tool_use_with_thought_signature():
    """Test that ToolUse can include optional thoughtSignature field."""
    tool_use: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {"key": "value"},
        "thoughtSignature": "YWJjZGVmZ2g=",
    }

    assert tool_use["toolUseId"] == "test-id"
    assert tool_use["name"] == "test_tool"
    assert tool_use["input"] == {"key": "value"}
    assert tool_use["thoughtSignature"] == "YWJjZGVmZ2g="


def test_tool_use_thought_signature_is_optional():
    """Test that thoughtSignature is truly optional and doesn't require all fields."""
    # Create with thoughtSignature
    tool_use_with_sig: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {},
        "thoughtSignature": "test",
    }
    assert "thoughtSignature" in tool_use_with_sig

    # Create without thoughtSignature
    tool_use_without_sig: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {},
    }
    assert "thoughtSignature" not in tool_use_without_sig


def test_tool_use_empty_input():
    """Test that ToolUse works with empty input."""
    tool_use: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {},
    }

    assert tool_use["input"] == {}
    assert "thoughtSignature" not in tool_use


def test_tool_use_complex_input():
    """Test that ToolUse works with complex nested input."""
    tool_use: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {
            "nested": {"key": "value"},
            "array": [1, 2, 3],
            "string": "test",
        },
        "thoughtSignature": "c2lnbmF0dXJl",
    }

    assert tool_use["input"]["nested"]["key"] == "value"
    assert tool_use["input"]["array"] == [1, 2, 3]
    assert tool_use["thoughtSignature"] == "c2lnbmF0dXJl"


def test_tool_use_base64_encoded_signature():
    """Test that thoughtSignature should be base64 encoded string."""
    # Valid base64 encoded signature
    tool_use: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {},
        "thoughtSignature": "dGVzdF9zaWduYXR1cmVfYnl0ZXM=",
    }

    assert tool_use["thoughtSignature"] == "dGVzdF9zaWduYXR1cmVfYnl0ZXM="

    # Empty signature should also be valid
    tool_use_empty: ToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "input": {},
        "thoughtSignature": "",
    }

    assert tool_use_empty["thoughtSignature"] == ""

