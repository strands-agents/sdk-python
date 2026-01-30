"""Tests for model validation helper functions."""

from strands.models._validation import _has_location_source


class TestHasLocationSource:
    """Tests for _has_location_source helper function."""

    def test_image_with_location_source(self):
        """Test detection of location source in image content."""
        content = {"image": {"source": {"location": {"type": "s3", "uri": "s3://bucket/key"}}}}
        assert _has_location_source(content)

    def test_image_with_bytes_source(self):
        """Test that bytes source is not detected as location."""
        content = {"image": {"source": {"bytes": b"data"}}}
        assert not _has_location_source(content)

    def test_document_with_location_source(self):
        """Test detection of location source in document content."""
        content = {"document": {"source": {"location": {"type": "s3", "uri": "s3://bucket/key"}}}}
        assert _has_location_source(content)

    def test_document_with_bytes_source(self):
        """Test that bytes source is not detected as location."""
        content = {"document": {"source": {"bytes": b"data"}}}
        assert not _has_location_source(content)

    def test_video_with_location_source(self):
        """Test detection of location source in video content."""
        content = {"video": {"source": {"location": {"type": "s3", "uri": "s3://bucket/key"}}}}
        assert _has_location_source(content)

    def test_video_with_bytes_source(self):
        """Test that bytes source is not detected as location."""
        content = {"video": {"source": {"bytes": b"data"}}}
        assert not _has_location_source(content)

    def test_text_content(self):
        """Test that text content is not detected as location source."""
        content = {"text": "hello"}
        assert not _has_location_source(content)

    def test_tool_use_content(self):
        """Test that toolUse content is not detected as location source."""
        content = {"toolUse": {"name": "test", "input": {}, "toolUseId": "123"}}
        assert not _has_location_source(content)

    def test_tool_result_content(self):
        """Test that toolResult content is not detected as location source."""
        content = {"toolResult": {"toolUseId": "123", "content": [{"text": "result"}]}}
        assert not _has_location_source(content)

    def test_image_without_source(self):
        """Test that image without source is not detected as location."""
        content = {"image": {"format": "png"}}
        assert not _has_location_source(content)

    def test_document_without_source(self):
        """Test that document without source is not detected as location."""
        content = {"document": {"format": "pdf", "name": "test.pdf"}}
        assert not _has_location_source(content)

    def test_video_without_source(self):
        """Test that video without source is not detected as location."""
        content = {"video": {"format": "mp4"}}
        assert not _has_location_source(content)
