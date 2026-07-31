"""Tests for media type definitions."""

from strands.types.media import (
    DocumentBlockContent,
    DocumentSource,
    ImageSource,
    S3Location,
    VideoSource,
)


class TestS3Location:
    """Tests for S3Location TypedDict."""

    def test_s3_location_with_uri_only(self):
        """Test S3Location with only uri field."""
        s3_loc: S3Location = {"uri": "s3://my-bucket/path/to/file.pdf"}

        assert s3_loc["uri"] == "s3://my-bucket/path/to/file.pdf"
        assert "bucketOwner" not in s3_loc

    def test_s3_location_with_bucket_owner(self):
        """Test S3Location with both uri and bucketOwner fields."""
        s3_loc: S3Location = {
            "uri": "s3://my-bucket/path/to/file.pdf",
            "bucketOwner": "123456789012",
        }

        assert s3_loc["uri"] == "s3://my-bucket/path/to/file.pdf"
        assert s3_loc["bucketOwner"] == "123456789012"


class TestDocumentSource:
    """Tests for DocumentSource TypedDict."""

    def test_document_source_with_bytes(self):
        """Test DocumentSource with bytes content."""
        doc_source: DocumentSource = {"bytes": b"document content"}

        assert doc_source["bytes"] == b"document content"
        assert "s3Location" not in doc_source

    def test_document_source_with_s3_location(self):
        """Test DocumentSource with s3Location."""
        doc_source: DocumentSource = {
            "s3Location": {
                "uri": "s3://my-bucket/docs/report.pdf",
                "bucketOwner": "123456789012",
            }
        }

        assert "bytes" not in doc_source
        assert doc_source["s3Location"]["uri"] == "s3://my-bucket/docs/report.pdf"
        assert doc_source["s3Location"]["bucketOwner"] == "123456789012"

    def test_document_source_with_text(self):
        """Test DocumentSource with text content."""
        doc_source: DocumentSource = {"text": "plain text content"}

        assert doc_source["text"] == "plain text content"
        assert "bytes" not in doc_source
        assert "location" not in doc_source
        assert "content" not in doc_source

    def test_document_source_with_content(self):
        """Test DocumentSource with content blocks."""
        doc_source: DocumentSource = {"content": [{"text": "block one"}, {"text": "block two"}]}

        assert len(doc_source["content"]) == 2
        assert doc_source["content"][0]["text"] == "block one"
        assert doc_source["content"][1]["text"] == "block two"
        assert "bytes" not in doc_source
        assert "location" not in doc_source
        assert "text" not in doc_source


class TestDocumentBlockContent:
    """Tests for DocumentBlockContent TypedDict."""

    def test_document_block_content_with_text(self):
        """Test DocumentBlockContent with text field."""
        block: DocumentBlockContent = {"text": "hello"}

        assert block["text"] == "hello"

    def test_document_block_content_empty(self):
        """Test DocumentBlockContent with no fields (total=False)."""
        block: DocumentBlockContent = {}

        assert "text" not in block


class TestImageSource:
    """Tests for ImageSource TypedDict."""

    def test_image_source_with_bytes(self):
        """Test ImageSource with bytes content."""
        img_source: ImageSource = {"bytes": b"image content"}

        assert img_source["bytes"] == b"image content"
        assert "s3Location" not in img_source

    def test_image_source_with_s3_location(self):
        """Test ImageSource with s3Location."""
        img_source: ImageSource = {
            "s3Location": {
                "uri": "s3://my-bucket/images/photo.png",
            }
        }

        assert "bytes" not in img_source
        assert img_source["s3Location"]["uri"] == "s3://my-bucket/images/photo.png"


class TestVideoSource:
    """Tests for VideoSource TypedDict."""

    def test_video_source_with_bytes(self):
        """Test VideoSource with bytes content."""
        vid_source: VideoSource = {"bytes": b"video content"}

        assert vid_source["bytes"] == b"video content"
        assert "s3Location" not in vid_source

    def test_video_source_with_s3_location(self):
        """Test VideoSource with s3Location."""
        vid_source: VideoSource = {
            "s3Location": {
                "uri": "s3://my-bucket/videos/clip.mp4",
                "bucketOwner": "987654321098",
            }
        }

        assert "bytes" not in vid_source
        assert vid_source["s3Location"]["uri"] == "s3://my-bucket/videos/clip.mp4"
        assert vid_source["s3Location"]["bucketOwner"] == "987654321098"
