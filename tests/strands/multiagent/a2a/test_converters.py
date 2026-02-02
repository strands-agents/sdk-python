"""Tests for A2A converter functions."""

import base64
import logging
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from a2a.types import (
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.types import Message as A2AMessage

from strands.agent.agent_result import AgentResult
from strands.multiagent.a2a._converters import (
    _convert_document_to_file_part,
    _convert_file_part_to_content_block,
    _convert_image_to_file_part,
    _convert_video_to_file_part,
    _get_location_from_uri,
    convert_content_blocks_to_parts,
    convert_input_to_message,
    convert_response_to_agent_result,
)

# --- Input conversion tests (Strands → A2A) ---


def test_convert_string_input():
    """Test converting string input to A2A message."""
    message = convert_input_to_message("Hello")

    assert isinstance(message, A2AMessage)
    assert message.role == Role.user
    assert len(message.parts) == 1
    assert message.parts[0].root.text == "Hello"


def test_convert_message_list_input():
    """Test converting message list input to A2A message."""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    message = convert_input_to_message(messages)

    assert isinstance(message, A2AMessage)
    assert message.role == Role.user
    assert len(message.parts) == 1


def test_convert_content_blocks_input():
    """Test converting content blocks input to A2A message."""
    content_blocks = [{"text": "Hello"}, {"text": "World"}]

    message = convert_input_to_message(content_blocks)

    assert isinstance(message, A2AMessage)
    assert len(message.parts) == 2


def test_convert_unsupported_input():
    """Test that unsupported input types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported input type"):
        convert_input_to_message(123)


def test_convert_interrupt_response_raises_error():
    """Test that InterruptResponseContent raises explicit error."""
    interrupt_responses = [{"interruptResponse": {"interruptId": "123", "response": "A"}}]

    with pytest.raises(ValueError, match="InterruptResponseContent is not supported for A2AAgent"):
        convert_input_to_message(interrupt_responses)


def test_convert_content_blocks_to_parts():
    """Test converting content blocks to A2A parts."""
    content_blocks = [{"text": "Hello"}, {"text": "World"}]

    parts = convert_content_blocks_to_parts(content_blocks)

    assert len(parts) == 2
    assert parts[0].root.text == "Hello"
    assert parts[1].root.text == "World"


def test_convert_a2a_message_response():
    """Test converting A2A message response to AgentResult."""
    a2a_message = A2AMessage(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[Part(TextPart(kind="text", text="Response"))],
    )

    result = convert_response_to_agent_result(a2a_message)

    assert isinstance(result, AgentResult)
    assert result.message["role"] == "assistant"
    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Response"


def test_convert_task_response():
    """Test converting task response to AgentResult."""
    mock_task = MagicMock()
    mock_artifact = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Task response"
    mock_artifact.parts = [mock_part]
    mock_task.artifacts = [mock_artifact]

    result = convert_response_to_agent_result((mock_task, None))

    assert isinstance(result, AgentResult)
    assert len(result.message["content"]) == 1
    assert result.message["content"][0]["text"] == "Task response"


def test_convert_multiple_parts_response():
    """Test converting response with multiple parts to separate content blocks."""
    a2a_message = A2AMessage(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[
            Part(TextPart(kind="text", text="First")),
            Part(TextPart(kind="text", text="Second")),
        ],
    )

    result = convert_response_to_agent_result(a2a_message)

    assert len(result.message["content"]) == 2
    assert result.message["content"][0]["text"] == "First"
    assert result.message["content"][1]["text"] == "Second"


def test_convert_message_list_finds_last_user_message():
    """Test that message list conversion finds the last user message."""
    messages = [
        {"role": "user", "content": [{"text": "First"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
        {"role": "user", "content": [{"text": "Second"}]},
    ]

    message = convert_input_to_message(messages)

    assert message.parts[0].root.text == "Second"


def test_convert_content_blocks_skips_non_text():
    """Test that non-text content blocks are skipped."""
    content_blocks = [{"text": "Hello"}, {"toolUse": "data"}, {"text": "World"}]

    parts = convert_content_blocks_to_parts(content_blocks)

    assert len(parts) == 2


def test_convert_task_artifact_update_event():
    """Test converting TaskArtifactUpdateEvent to AgentResult."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Streamed artifact"
    mock_artifact = MagicMock()
    mock_artifact.parts = [mock_part]

    mock_event = MagicMock(spec=TaskArtifactUpdateEvent)
    mock_event.artifact = mock_artifact

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert result.message["content"][0]["text"] == "Streamed artifact"


def test_convert_task_status_update_event():
    """Test converting TaskStatusUpdateEvent to AgentResult."""
    mock_task = MagicMock()
    mock_part = MagicMock()
    mock_part.root.text = "Status message"
    mock_message = MagicMock()
    mock_message.parts = [mock_part]
    mock_status = MagicMock()
    mock_status.message = mock_message

    mock_event = MagicMock(spec=TaskStatusUpdateEvent)
    mock_event.status = mock_status

    result = convert_response_to_agent_result((mock_task, mock_event))

    assert result.message["content"][0]["text"] == "Status message"


def test_convert_response_handles_missing_data():
    """Test that response conversion handles missing/malformed data gracefully."""
    # TaskArtifactUpdateEvent with no artifact
    mock_event = MagicMock(spec=TaskArtifactUpdateEvent)
    mock_event.artifact = None
    result = convert_response_to_agent_result((MagicMock(), mock_event))
    assert len(result.message["content"]) == 0

    # TaskStatusUpdateEvent with no status
    mock_event = MagicMock(spec=TaskStatusUpdateEvent)
    mock_event.status = None
    result = convert_response_to_agent_result((MagicMock(), mock_event))
    assert len(result.message["content"]) == 0

    # Task artifact without parts attribute
    mock_task = MagicMock()
    mock_artifact = MagicMock(spec=[])
    del mock_artifact.parts
    mock_task.artifacts = [mock_artifact]
    result = convert_response_to_agent_result((mock_task, None))
    assert len(result.message["content"]) == 0


# --- URI type handling tests ---


class TestUriTypeHandling:
    """Tests for URI type detection and location creation."""

    def test_get_location_from_s3_uri(self):
        """Test that S3 URIs get type 's3'."""
        location = _get_location_from_uri("s3://bucket/path/file.png")

        assert location["type"] == "s3"
        assert location["uri"] == "s3://bucket/path/file.png"

    def test_get_location_from_http_uri(self):
        """Test that HTTP URIs get type 'url'."""
        location = _get_location_from_uri("http://example.com/image.png")

        assert location["type"] == "url"
        assert location["uri"] == "http://example.com/image.png"

    def test_get_location_from_https_uri(self):
        """Test that HTTPS URIs get type 'url'."""
        location = _get_location_from_uri("https://cdn.example.com/assets/image.jpg")

        assert location["type"] == "url"
        assert location["uri"] == "https://cdn.example.com/assets/image.jpg"

    def test_get_location_from_unknown_scheme(self):
        """Test that unknown URI schemes get type 'uri'."""
        location = _get_location_from_uri("ftp://server/file.pdf")

        assert location["type"] == "uri"
        assert location["uri"] == "ftp://server/file.pdf"

    def test_convert_file_part_with_http_uri_to_image(self):
        """Test converting A2A FilePart with HTTP URI to Strands ImageContent."""
        file_with_uri = FileWithUri(uri="https://example.com/image.png", mime_type="image/png")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "image" in block
        assert block["image"]["source"]["location"]["type"] == "url"
        assert block["image"]["source"]["location"]["uri"] == "https://example.com/image.png"

    def test_convert_file_part_with_s3_uri_to_image(self):
        """Test converting A2A FilePart with S3 URI to Strands ImageContent."""
        file_with_uri = FileWithUri(uri="s3://bucket/image.png", mime_type="image/png")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "image" in block
        assert block["image"]["source"]["location"]["type"] == "s3"
        assert block["image"]["source"]["location"]["uri"] == "s3://bucket/image.png"

    def test_convert_file_part_with_http_uri_to_document(self):
        """Test converting A2A FilePart with HTTP URI to Strands DocumentContent."""
        file_with_uri = FileWithUri(uri="https://example.com/doc.pdf", mime_type="application/pdf", name="doc.pdf")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "document" in block
        assert block["document"]["source"]["location"]["type"] == "url"
        assert block["document"]["source"]["location"]["uri"] == "https://example.com/doc.pdf"

    def test_convert_file_part_with_http_uri_to_video(self):
        """Test converting A2A FilePart with HTTP URI to Strands VideoContent."""
        file_with_uri = FileWithUri(uri="https://cdn.example.com/video.mp4", mime_type="video/mp4")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "video" in block
        assert block["video"]["source"]["location"]["type"] == "url"
        assert block["video"]["source"]["location"]["uri"] == "https://cdn.example.com/video.mp4"

    def test_convert_image_with_http_location_to_file_part(self):
        """Test converting Strands image with HTTP location to A2A FilePart."""
        image_content = {
            "format": "png",
            "source": {"location": {"type": "url", "uri": "https://example.com/image.png"}},
        }

        part = _convert_image_to_file_part(image_content)

        assert part is not None
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "https://example.com/image.png"


# --- Logging tests for dropped content ---


class TestLoggingForDroppedContent:
    """Tests for debug logging when content is dropped."""

    def test_image_empty_source_logs_debug(self, caplog):
        """Test that dropping image with empty source logs debug message."""
        image_content = {"format": "png", "source": {}}

        with caplog.at_level(logging.DEBUG, logger="strands.multiagent.a2a._converters"):
            result = _convert_image_to_file_part(image_content)

        assert result is None
        assert "image content dropped" in caplog.text

    def test_document_empty_source_logs_debug(self, caplog):
        """Test that dropping document with empty source logs debug message."""
        doc_content = {"format": "pdf", "name": "test.pdf", "source": {}}

        with caplog.at_level(logging.DEBUG, logger="strands.multiagent.a2a._converters"):
            result = _convert_document_to_file_part(doc_content)

        assert result is None
        assert "document content dropped" in caplog.text

    def test_video_empty_source_logs_debug(self, caplog):
        """Test that dropping video with empty source logs debug message."""
        video_content = {"format": "mp4", "source": {}}

        with caplog.at_level(logging.DEBUG, logger="strands.multiagent.a2a._converters"):
            result = _convert_video_to_file_part(video_content)

        assert result is None
        assert "video content dropped" in caplog.text

    def test_unsupported_mime_type_logs_debug(self, caplog):
        """Test that dropping file part with unsupported MIME type logs debug message."""
        file_with_bytes = FileWithBytes(bytes="dGVzdA==", mime_type="audio/mp3")
        file_part = FilePart(file=file_with_bytes, kind="file")

        with caplog.at_level(logging.DEBUG, logger="strands.multiagent.a2a._converters"):
            result = _convert_file_part_to_content_block(file_part)

        assert result is None
        assert "file part dropped" in caplog.text
        assert "audio/mp3" in caplog.text


# --- Image content conversion tests ---


class TestImageConversion:
    """Tests for image content type conversion."""

    def test_convert_image_with_bytes_to_file_part(self):
        """Test converting Strands image with inline bytes to A2A FilePart."""
        image_bytes = b"fake png data"
        image_content = {
            "format": "png",
            "source": {"bytes": image_bytes},
        }

        part = _convert_image_to_file_part(image_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        assert part.root.file.mime_type == "image/png"
        assert isinstance(part.root.file, FileWithBytes)
        assert base64.standard_b64decode(part.root.file.bytes) == image_bytes

    def test_convert_image_with_s3_location_to_file_part(self):
        """Test converting Strands image with S3 location to A2A FilePart."""
        image_content = {
            "format": "jpeg",
            "source": {"location": {"type": "s3", "uri": "s3://bucket/image.jpg"}},
        }

        part = _convert_image_to_file_part(image_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        assert part.root.file.mime_type == "image/jpeg"
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "s3://bucket/image.jpg"

    def test_convert_image_returns_none_for_empty_source(self):
        """Test that image conversion returns None when source is empty."""
        image_content = {"format": "png", "source": {}}

        part = _convert_image_to_file_part(image_content)

        assert part is None

    def test_convert_image_all_formats(self):
        """Test conversion of all supported image formats."""
        formats_and_mimes = [
            ("png", "image/png"),
            ("jpeg", "image/jpeg"),
            ("gif", "image/gif"),
            ("webp", "image/webp"),
        ]

        for fmt, expected_mime in formats_and_mimes:
            image_content = {"format": fmt, "source": {"bytes": b"data"}}
            part = _convert_image_to_file_part(image_content)
            assert part is not None
            assert part.root.file.mime_type == expected_mime

    def test_convert_content_blocks_with_image(self):
        """Test converting content blocks containing images."""
        content_blocks = [
            {"text": "Here is an image:"},
            {"image": {"format": "png", "source": {"bytes": b"png data"}}},
        ]

        parts = convert_content_blocks_to_parts(content_blocks)

        assert len(parts) == 2
        assert parts[0].root.text == "Here is an image:"
        assert isinstance(parts[1].root, FilePart)
        assert parts[1].root.file.mime_type == "image/png"


class TestImageOutputConversion:
    """Tests for A2A FilePart to Strands image conversion."""

    def test_convert_file_part_to_image_with_bytes(self):
        """Test converting A2A FilePart with bytes to Strands ImageContent."""
        raw_bytes = b"image data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="image/png")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "image" in block
        assert block["image"]["format"] == "png"
        assert block["image"]["source"]["bytes"] == raw_bytes

    def test_convert_file_part_to_image_with_uri(self):
        """Test converting A2A FilePart with URI to Strands ImageContent."""
        file_with_uri = FileWithUri(uri="s3://bucket/image.jpeg", mime_type="image/jpeg")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "image" in block
        assert block["image"]["format"] == "jpeg"
        assert block["image"]["source"]["location"]["uri"] == "s3://bucket/image.jpeg"

    def test_convert_response_with_image_file_part(self):
        """Test full response conversion with image FilePart."""
        raw_bytes = b"image content"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="image/png")
        file_part = FilePart(file=file_with_bytes, kind="file")

        a2a_message = A2AMessage(
            message_id=uuid4().hex,
            role=Role.agent,
            parts=[Part(file_part)],
        )

        result = convert_response_to_agent_result(a2a_message)

        assert len(result.message["content"]) == 1
        assert "image" in result.message["content"][0]
        assert result.message["content"][0]["image"]["source"]["bytes"] == raw_bytes


# --- Document content conversion tests ---


class TestDocumentConversion:
    """Tests for document content type conversion."""

    def test_convert_document_with_bytes_to_file_part(self):
        """Test converting Strands document with inline bytes to A2A FilePart."""
        doc_bytes = b"%PDF-1.4 fake pdf content"
        doc_content = {
            "format": "pdf",
            "name": "report.pdf",
            "source": {"bytes": doc_bytes},
        }

        part = _convert_document_to_file_part(doc_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        assert part.root.file.mime_type == "application/pdf"
        assert isinstance(part.root.file, FileWithBytes)
        assert base64.standard_b64decode(part.root.file.bytes) == doc_bytes
        assert part.root.file.name == "report.pdf"

    def test_convert_document_with_s3_location_to_file_part(self):
        """Test converting Strands document with S3 location to A2A FilePart."""
        doc_content = {
            "format": "docx",
            "name": "document.docx",
            "source": {"location": {"type": "s3", "uri": "s3://bucket/doc.docx"}},
        }

        part = _convert_document_to_file_part(doc_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        expected_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert part.root.file.mime_type == expected_mime
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "s3://bucket/doc.docx"
        assert part.root.file.name == "document.docx"

    def test_convert_document_returns_none_for_empty_source(self):
        """Test that document conversion returns None when source is empty."""
        doc_content = {"format": "pdf", "source": {}}

        part = _convert_document_to_file_part(doc_content)

        assert part is None

    def test_convert_document_all_formats(self):
        """Test conversion of all supported document formats."""
        formats_and_mimes = [
            ("pdf", "application/pdf"),
            ("csv", "text/csv"),
            ("doc", "application/msword"),
            ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("xls", "application/vnd.ms-excel"),
            ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("html", "text/html"),
            ("txt", "text/plain"),
            ("md", "text/markdown"),
        ]

        for fmt, expected_mime in formats_and_mimes:
            doc_content = {"format": fmt, "source": {"bytes": b"data"}}
            part = _convert_document_to_file_part(doc_content)
            assert part is not None, f"Failed for format {fmt}"
            assert part.root.file.mime_type == expected_mime, f"MIME mismatch for format {fmt}"

    def test_convert_content_blocks_with_document(self):
        """Test converting content blocks containing documents."""
        content_blocks = [
            {"text": "Here is a document:"},
            {"document": {"format": "pdf", "name": "test.pdf", "source": {"bytes": b"pdf data"}}},
        ]

        parts = convert_content_blocks_to_parts(content_blocks)

        assert len(parts) == 2
        assert parts[0].root.text == "Here is a document:"
        assert isinstance(parts[1].root, FilePart)
        assert parts[1].root.file.mime_type == "application/pdf"
        assert parts[1].root.file.name == "test.pdf"


class TestDocumentOutputConversion:
    """Tests for A2A FilePart to Strands document conversion."""

    def test_convert_file_part_to_document_with_bytes(self):
        """Test converting A2A FilePart with bytes to Strands DocumentContent."""
        raw_bytes = b"document data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="application/pdf", name="test.pdf")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "document" in block
        assert block["document"]["format"] == "pdf"
        assert block["document"]["source"]["bytes"] == raw_bytes
        assert block["document"]["name"] == "test.pdf"

    def test_convert_file_part_to_document_with_uri(self):
        """Test converting A2A FilePart with URI to Strands DocumentContent."""
        file_with_uri = FileWithUri(uri="s3://bucket/doc.csv", mime_type="text/csv", name="data.csv")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "document" in block
        assert block["document"]["format"] == "csv"
        assert block["document"]["source"]["location"]["uri"] == "s3://bucket/doc.csv"
        assert block["document"]["name"] == "data.csv"

    def test_convert_response_with_document_file_part(self):
        """Test full response conversion with document FilePart."""
        raw_bytes = b"document content"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="application/pdf", name="report.pdf")
        file_part = FilePart(file=file_with_bytes, kind="file")

        a2a_message = A2AMessage(
            message_id=uuid4().hex,
            role=Role.agent,
            parts=[Part(file_part)],
        )

        result = convert_response_to_agent_result(a2a_message)

        assert len(result.message["content"]) == 1
        assert "document" in result.message["content"][0]
        assert result.message["content"][0]["document"]["source"]["bytes"] == raw_bytes


# --- Video content conversion tests ---


class TestVideoConversion:
    """Tests for video content type conversion."""

    def test_convert_video_with_bytes_to_file_part(self):
        """Test converting Strands video with inline bytes to A2A FilePart."""
        video_bytes = b"fake mp4 video data"
        video_content = {
            "format": "mp4",
            "source": {"bytes": video_bytes},
        }

        part = _convert_video_to_file_part(video_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        assert part.root.file.mime_type == "video/mp4"
        assert isinstance(part.root.file, FileWithBytes)
        assert base64.standard_b64decode(part.root.file.bytes) == video_bytes

    def test_convert_video_with_s3_location_to_file_part(self):
        """Test converting Strands video with S3 location to A2A FilePart."""
        video_content = {
            "format": "webm",
            "source": {"location": {"type": "s3", "uri": "s3://bucket/video.webm"}},
        }

        part = _convert_video_to_file_part(video_content)

        assert part is not None
        assert isinstance(part.root, FilePart)
        assert part.root.file.mime_type == "video/webm"
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "s3://bucket/video.webm"

    def test_convert_video_returns_none_for_empty_source(self):
        """Test that video conversion returns None when source is empty."""
        video_content = {"format": "mp4", "source": {}}

        part = _convert_video_to_file_part(video_content)

        assert part is None

    def test_convert_video_all_formats(self):
        """Test conversion of all supported video formats."""
        formats_and_mimes = [
            ("flv", "video/x-flv"),
            ("mkv", "video/x-matroska"),
            ("mov", "video/quicktime"),
            ("mpeg", "video/mpeg"),
            ("mpg", "video/mpeg"),
            ("mp4", "video/mp4"),
            ("three_gp", "video/3gpp"),
            ("webm", "video/webm"),
            ("wmv", "video/x-ms-wmv"),
        ]

        for fmt, expected_mime in formats_and_mimes:
            video_content = {"format": fmt, "source": {"bytes": b"data"}}
            part = _convert_video_to_file_part(video_content)
            assert part is not None, f"Failed for format {fmt}"
            assert part.root.file.mime_type == expected_mime, f"MIME mismatch for format {fmt}"

    def test_convert_content_blocks_with_video(self):
        """Test converting content blocks containing videos."""
        content_blocks = [
            {"text": "Here is a video:"},
            {"video": {"format": "mp4", "source": {"bytes": b"video data"}}},
        ]

        parts = convert_content_blocks_to_parts(content_blocks)

        assert len(parts) == 2
        assert parts[0].root.text == "Here is a video:"
        assert isinstance(parts[1].root, FilePart)
        assert parts[1].root.file.mime_type == "video/mp4"


class TestVideoOutputConversion:
    """Tests for A2A FilePart to Strands video conversion."""

    def test_convert_file_part_to_video_with_bytes(self):
        """Test converting A2A FilePart with bytes to Strands VideoContent."""
        raw_bytes = b"video data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="video/mp4")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "video" in block
        assert block["video"]["format"] == "mp4"
        assert block["video"]["source"]["bytes"] == raw_bytes

    def test_convert_file_part_to_video_with_uri(self):
        """Test converting A2A FilePart with URI to Strands VideoContent."""
        file_with_uri = FileWithUri(uri="s3://bucket/video.webm", mime_type="video/webm")
        file_part = FilePart(file=file_with_uri, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "video" in block
        assert block["video"]["format"] == "webm"
        assert block["video"]["source"]["location"]["uri"] == "s3://bucket/video.webm"

    def test_convert_response_with_video_file_part(self):
        """Test full response conversion with video FilePart."""
        raw_bytes = b"video content"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="video/mp4")
        file_part = FilePart(file=file_with_bytes, kind="file")

        a2a_message = A2AMessage(
            message_id=uuid4().hex,
            role=Role.agent,
            parts=[Part(file_part)],
        )

        result = convert_response_to_agent_result(a2a_message)

        assert len(result.message["content"]) == 1
        assert "video" in result.message["content"][0]
        assert result.message["content"][0]["video"]["source"]["bytes"] == raw_bytes


# --- Mixed content and edge case tests ---


class TestMixedContentConversion:
    """Tests for mixed content type conversion."""

    def test_convert_content_blocks_with_all_types(self):
        """Test converting content blocks with text, image, document, and video."""
        content_blocks = [
            {"text": "Introduction"},
            {"image": {"format": "png", "source": {"bytes": b"image"}}},
            {"document": {"format": "pdf", "name": "doc.pdf", "source": {"bytes": b"doc"}}},
            {"video": {"format": "mp4", "source": {"bytes": b"video"}}},
            {"text": "Conclusion"},
        ]

        parts = convert_content_blocks_to_parts(content_blocks)

        assert len(parts) == 5
        assert parts[0].root.text == "Introduction"
        assert isinstance(parts[1].root, FilePart)
        assert parts[1].root.file.mime_type == "image/png"
        assert isinstance(parts[2].root, FilePart)
        assert parts[2].root.file.mime_type == "application/pdf"
        assert isinstance(parts[3].root, FilePart)
        assert parts[3].root.file.mime_type == "video/mp4"
        assert parts[4].root.text == "Conclusion"

    def test_convert_response_with_mixed_parts(self):
        """Test response conversion with mixed text and file parts."""
        raw_bytes = b"image data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="image/png")
        file_part = FilePart(file=file_with_bytes, kind="file")

        a2a_message = A2AMessage(
            message_id=uuid4().hex,
            role=Role.agent,
            parts=[
                Part(TextPart(kind="text", text="Here is an image:")),
                Part(file_part),
            ],
        )

        result = convert_response_to_agent_result(a2a_message)

        assert len(result.message["content"]) == 2
        assert result.message["content"][0]["text"] == "Here is an image:"
        assert "image" in result.message["content"][1]

    def test_convert_file_part_with_unknown_mime_type(self):
        """Test that unknown MIME types return None."""
        file_with_bytes = FileWithBytes(bytes="dGVzdA==", mime_type="application/x-unknown")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        # application/* falls back to document conversion
        assert block is not None
        assert "document" in block

    def test_convert_file_part_with_generic_image_mime(self):
        """Test that generic image/* MIME types are handled as images."""
        raw_bytes = b"image data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="image/x-custom")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "image" in block
        # Falls back to png format for unknown image types
        assert block["image"]["format"] == "png"

    def test_convert_file_part_with_generic_video_mime(self):
        """Test that generic video/* MIME types are handled as videos."""
        raw_bytes = b"video data"
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type="video/x-custom")
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        assert block is not None
        assert "video" in block
        # Falls back to mp4 format for unknown video types
        assert block["video"]["format"] == "mp4"

    def test_convert_file_part_with_no_mime_type(self):
        """Test conversion when MIME type is missing."""
        file_with_bytes = FileWithBytes(bytes="dGVzdA==")  # No mime_type specified
        file_part = FilePart(file=file_with_bytes, kind="file")

        block = _convert_file_part_to_content_block(file_part)

        # Defaults to application/octet-stream which falls back to document
        assert block is not None
        assert "document" in block

    def test_convert_input_message_with_image(self):
        """Test converting input message containing images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": "What is in this image?"},
                    {"image": {"format": "jpeg", "source": {"bytes": b"jpeg data"}}},
                ],
            }
        ]

        message = convert_input_to_message(messages)

        assert len(message.parts) == 2
        assert message.parts[0].root.text == "What is in this image?"
        assert isinstance(message.parts[1].root, FilePart)
        assert message.parts[1].root.file.mime_type == "image/jpeg"
