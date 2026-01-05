"""Tests for content-related type definitions."""


class TestGuardrailConverseImageSource:
    """Tests for GuardrailConverseImageSource TypedDict."""

    def test_structure_with_bytes(self):
        """Test GuardrailConverseImageSource structure accepts bytes."""
        from strands.types.content import GuardrailConverseImageSource

        image_bytes = b"fake image data"
        source: GuardrailConverseImageSource = {"bytes": image_bytes}

        assert source["bytes"] == image_bytes
        assert isinstance(source["bytes"], bytes)


class TestGuardContentImage:
    """Tests for GuardContentImage TypedDict."""

    def test_structure_with_png_format(self):
        """Test GuardContentImage with png format."""
        from strands.types.content import GuardContentImage

        image_bytes = b"fake png data"
        image: GuardContentImage = {"format": "png", "source": {"bytes": image_bytes}}

        assert image["format"] == "png"
        assert image["source"]["bytes"] == image_bytes

    def test_structure_with_jpeg_format(self):
        """Test GuardContentImage with jpeg format."""
        from strands.types.content import GuardContentImage

        image_bytes = b"fake jpeg data"
        image: GuardContentImage = {"format": "jpeg", "source": {"bytes": image_bytes}}

        assert image["format"] == "jpeg"
        assert image["source"]["bytes"] == image_bytes


class TestGuardContent:
    """Tests for GuardContent TypedDict with image support."""

    def test_with_text_only(self):
        """Test GuardContent with text field only (existing behavior)."""
        from strands.types.content import GuardContent

        guard: GuardContent = {"text": {"text": "sample text", "qualifiers": ["query"]}}

        assert guard["text"]["text"] == "sample text"
        assert guard["text"]["qualifiers"] == ["query"]

    def test_with_image_only(self):
        """Test GuardContent with image field only."""
        from strands.types.content import GuardContent

        image_bytes = b"fake image data"
        guard: GuardContent = {"image": {"format": "png", "source": {"bytes": image_bytes}}}

        assert guard["image"]["format"] == "png"
        assert guard["image"]["source"]["bytes"] == image_bytes

    def test_with_both_text_and_image(self):
        """Test GuardContent with both text and image fields."""
        from strands.types.content import GuardContent

        image_bytes = b"fake image data"
        guard: GuardContent = {
            "text": {"text": "sample text", "qualifiers": ["query"]},
            "image": {"format": "jpeg", "source": {"bytes": image_bytes}},
        }

        assert guard["text"]["text"] == "sample text"
        assert guard["image"]["format"] == "jpeg"
        assert guard["image"]["source"]["bytes"] == image_bytes

    def test_optional_fields(self):
        """Test that GuardContent fields are optional."""
        from strands.types.content import GuardContent

        # Empty guard should be valid (all fields optional)
        guard: GuardContent = {}
        assert isinstance(guard, dict)
