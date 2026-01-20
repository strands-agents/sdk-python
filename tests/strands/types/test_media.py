"""Tests for media type definitions."""


from strands.types.media import (
    AudioContent,
    AudioSource,
    DocumentContent,
    ImageContent,
    VideoContent,
)


def test_audio_content_type():
    """Test AudioContent TypedDict creation."""
    audio: AudioContent = {
        "format": "mp3",
        "source": {"bytes": b"audio_data"},
    }
    assert audio["format"] == "mp3"
    assert audio["source"]["bytes"] == b"audio_data"


def test_audio_source_type():
    """Test AudioSource TypedDict creation."""
    source: AudioSource = {"bytes": b"test_audio"}
    assert source["bytes"] == b"test_audio"


def test_audio_format_literals():
    """Test that all expected audio formats are valid."""
    valid_formats = ["mp3", "wav", "flac", "ogg", "webm"]
    for fmt in valid_formats:
        # This would be a type error if fmt wasn't a valid AudioFormat
        audio: AudioContent = {
            "format": fmt,  # type: ignore  # Testing runtime behavior
            "source": {"bytes": b"data"},
        }
        assert audio["format"] == fmt


def test_image_content_type():
    """Test ImageContent TypedDict creation."""
    image: ImageContent = {
        "format": "png",
        "source": {"bytes": b"image_data"},
    }
    assert image["format"] == "png"
    assert image["source"]["bytes"] == b"image_data"


def test_video_content_type():
    """Test VideoContent TypedDict creation."""
    video: VideoContent = {
        "format": "mp4",
        "source": {"bytes": b"video_data"},
    }
    assert video["format"] == "mp4"
    assert video["source"]["bytes"] == b"video_data"


def test_document_content_type():
    """Test DocumentContent TypedDict creation."""
    doc: DocumentContent = {
        "format": "pdf",
        "name": "test.pdf",
        "source": {"bytes": b"pdf_data"},
    }
    assert doc["format"] == "pdf"
    assert doc["name"] == "test.pdf"
    assert doc["source"]["bytes"] == b"pdf_data"
