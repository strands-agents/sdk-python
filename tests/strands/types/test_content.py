"""Tests for content-related type definitions."""

import pytest

from strands.types.content import GuardContent, GuardContentImage, GuardrailConverseImageSource


@pytest.fixture
def image_bytes():
    """Provide sample image bytes for testing."""
    return b"fake image data"


@pytest.fixture
def sample_text():
    """Provide sample text content for GuardContent."""
    return {"text": "sample text", "qualifiers": ["query"]}


def test_structure_with_bytes(image_bytes):
    """Test GuardrailConverseImageSource structure accepts bytes."""
    source: GuardrailConverseImageSource = {"bytes": image_bytes}

    assert source["bytes"] == image_bytes
    assert isinstance(source["bytes"], bytes)


def test_structure_with_png_format(image_bytes):
    """Test GuardContentImage with png format."""
    image: GuardContentImage = {"format": "png", "source": {"bytes": image_bytes}}

    assert image["format"] == "png"
    assert image["source"]["bytes"] == image_bytes


def test_structure_with_jpeg_format(image_bytes):
    """Test GuardContentImage with jpeg format."""
    image: GuardContentImage = {"format": "jpeg", "source": {"bytes": image_bytes}}

    assert image["format"] == "jpeg"
    assert image["source"]["bytes"] == image_bytes


def test_with_text_only(sample_text):
    """Test GuardContent with text field only (existing behavior)."""
    guard: GuardContent = {"text": sample_text}

    assert guard["text"]["text"] == "sample text"
    assert guard["text"]["qualifiers"] == ["query"]


def test_with_image_only(image_bytes):
    """Test GuardContent with image field only."""
    guard: GuardContent = {"image": {"format": "png", "source": {"bytes": image_bytes}}}

    assert guard["image"]["format"] == "png"
    assert guard["image"]["source"]["bytes"] == image_bytes


def test_with_both_text_and_image(sample_text, image_bytes):
    """Test GuardContent with both text and image fields."""
    guard: GuardContent = {
        "text": sample_text,
        "image": {"format": "jpeg", "source": {"bytes": image_bytes}},
    }

    assert guard["text"]["text"] == "sample text"
    assert guard["image"]["format"] == "jpeg"
    assert guard["image"]["source"]["bytes"] == image_bytes
