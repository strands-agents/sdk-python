"""Tests for strands.types.content module."""


from strands.types.content import ContentBlockStartToolUse


def test_content_block_start_tool_use_required_fields():
    """Test that ContentBlockStartToolUse can be created with only required fields."""
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
    }

    assert content_block["toolUseId"] == "test-id"
    assert content_block["name"] == "test_tool"
    assert "thoughtSignature" not in content_block


def test_content_block_start_tool_use_with_thought_signature():
    """Test that ContentBlockStartToolUse can include optional thoughtSignature field."""
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "thoughtSignature": "YWJjZGVmZ2g=",
    }

    assert content_block["toolUseId"] == "test-id"
    assert content_block["name"] == "test_tool"
    assert content_block["thoughtSignature"] == "YWJjZGVmZ2g="


def test_content_block_start_tool_use_thought_signature_is_optional():
    """Test that thoughtSignature is truly optional."""
    # Create with thoughtSignature
    with_sig: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "thoughtSignature": "test",
    }
    assert "thoughtSignature" in with_sig

    # Create without thoughtSignature
    without_sig: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
    }
    assert "thoughtSignature" not in without_sig


def test_content_block_start_tool_use_base64_encoded():
    """Test that thoughtSignature should be base64 encoded string."""
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "thoughtSignature": "dGVzdF9zaWduYXR1cmVfYnl0ZXM=",
    }

    assert content_block["thoughtSignature"] == "dGVzdF9zaWduYXR1cmVfYnl0ZXM="


def test_content_block_start_tool_use_empty_signature():
    """Test that empty thoughtSignature is valid."""
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "thoughtSignature": "",
    }

    assert content_block["thoughtSignature"] == ""


def test_content_block_start_tool_use_special_characters_in_name():
    """Test that tool names with special characters work correctly."""
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "default_api:vehicleDetails",
        "thoughtSignature": "c2lnbmF0dXJl",
    }

    assert content_block["name"] == "default_api:vehicleDetails"
    assert content_block["thoughtSignature"] == "c2lnbmF0dXJl"


def test_content_block_start_tool_use_long_signature():
    """Test that long base64 encoded signatures are supported."""
    # Simulate a long signature (typical of Gemini's encrypted tokens)
    long_signature = "dGVzdF9zaWduYXR1cmVfYnl0ZXNfdGhhdF9pc192ZXJ5X2xvbmdf" * 5
    content_block: ContentBlockStartToolUse = {
        "toolUseId": "test-id",
        "name": "test_tool",
        "thoughtSignature": long_signature,
    }

    assert content_block["thoughtSignature"] == long_signature
    assert len(content_block["thoughtSignature"]) > 100
