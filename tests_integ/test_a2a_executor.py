"""Integration tests for A2A executor with real file processing."""

import base64
import os
from unittest.mock import MagicMock

import pytest
from a2a.types import FilePart, TextPart

from strands.multiagent.a2a.executor import StrandsA2AExecutor


@pytest.mark.asyncio
async def test_a2a_executor_with_real_image():
    """Test A2A executor processes a real image file correctly."""
    # Read the test image file
    test_image_path = os.path.join(os.path.dirname(__file__), "yellow.png")
    with open(test_image_path, "rb") as f:
        original_image_bytes = f.read()

    # Encode as base64 (A2A format)
    base64_image = base64.b64encode(original_image_bytes).decode("utf-8")

    # Create executor
    executor = StrandsA2AExecutor(MagicMock())

    # Create A2A message parts
    text_part = MagicMock(spec=TextPart)
    text_part.text = "Please analyze this image"
    text_part_mock = MagicMock()
    text_part_mock.root = text_part

    # Create file part with real image data
    file_obj = MagicMock()
    file_obj.name = "yellow.png"
    file_obj.mime_type = "image/png"
    file_obj.bytes = base64_image  # A2A sends base64-encoded string
    file_obj.uri = None

    file_part = MagicMock(spec=FilePart)
    file_part.file = file_obj
    file_part_mock = MagicMock()
    file_part_mock.root = file_part

    # Convert parts to content blocks
    parts = [text_part_mock, file_part_mock]
    content_blocks = executor._convert_a2a_parts_to_content_blocks(parts)

    # Verify conversion worked correctly
    assert len(content_blocks) == 2

    # Verify text conversion
    assert content_blocks[0]["text"] == "Please analyze this image"

    # Verify image conversion - most importantly, bytes should match original
    assert "image" in content_blocks[1]
    assert content_blocks[1]["image"]["format"] == "png"
    assert content_blocks[1]["image"]["source"]["bytes"] == original_image_bytes

    # Verify the round-trip: original -> base64 -> decoded == original
    assert len(content_blocks[1]["image"]["source"]["bytes"]) == len(original_image_bytes)
    assert content_blocks[1]["image"]["source"]["bytes"] == original_image_bytes


def test_a2a_executor_image_roundtrip():
    """Test that image data survives the A2A base64 encoding/decoding roundtrip."""
    # Read the test image
    test_image_path = os.path.join(os.path.dirname(__file__), "yellow.png")
    with open(test_image_path, "rb") as f:
        original_bytes = f.read()

    # Simulate A2A protocol: encode to base64 string
    base64_string = base64.b64encode(original_bytes).decode("utf-8")

    # Simulate executor decoding
    decoded_bytes = base64.b64decode(base64_string)

    # Verify perfect roundtrip
    assert decoded_bytes == original_bytes
    assert len(decoded_bytes) == len(original_bytes)

    # Verify it's actually image data (PNG signature)
    assert decoded_bytes.startswith(b"\x89PNG\r\n\x1a\n")
