"""Conversion functions between Strands and A2A types."""

import base64
from typing import cast
from uuid import uuid4

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

from ...agent.agent_result import AgentResult
from ...telemetry.metrics import EventLoopMetrics
from ...types.a2a import A2AResponse
from ...types.agent import AgentInput
from ...types.content import ContentBlock, Message
from ...types.media import (
    DocumentContent,
    DocumentFormat,
    ImageContent,
    ImageFormat,
    VideoContent,
    VideoFormat,
)

# MIME type mappings for Strands formats
IMAGE_FORMAT_TO_MIME: dict[ImageFormat, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

DOCUMENT_FORMAT_TO_MIME: dict[DocumentFormat, str] = {
    "pdf": "application/pdf",
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt": "text/plain",
    "md": "text/markdown",
}

VIDEO_FORMAT_TO_MIME: dict[VideoFormat, str] = {
    "flv": "video/x-flv",
    "mkv": "video/x-matroska",
    "mov": "video/quicktime",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
    "mp4": "video/mp4",
    "three_gp": "video/3gpp",
    "webm": "video/webm",
    "wmv": "video/x-ms-wmv",
}

# Reverse mappings from MIME type to Strands format
MIME_TO_IMAGE_FORMAT: dict[str, ImageFormat] = {v: k for k, v in IMAGE_FORMAT_TO_MIME.items()}
MIME_TO_DOCUMENT_FORMAT: dict[str, DocumentFormat] = {v: k for k, v in DOCUMENT_FORMAT_TO_MIME.items()}
MIME_TO_VIDEO_FORMAT: dict[str, VideoFormat] = {v: k for k, v in VIDEO_FORMAT_TO_MIME.items()}


def convert_input_to_message(prompt: AgentInput) -> A2AMessage:
    """Convert AgentInput to A2A Message.

    Args:
        prompt: Input in various formats (string, message list, or content blocks).

    Returns:
        A2AMessage ready to send to the remote agent.

    Raises:
        ValueError: If prompt format is unsupported.
    """
    message_id = uuid4().hex

    if isinstance(prompt, str):
        return A2AMessage(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(kind="text", text=prompt))],
            message_id=message_id,
        )

    if isinstance(prompt, list) and prompt and (isinstance(prompt[0], dict)):
        # Check for interrupt responses - not supported in A2A
        if "interruptResponse" in prompt[0]:
            raise ValueError("InterruptResponseContent is not supported for A2AAgent")

        if "role" in prompt[0]:
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    content = cast(list[ContentBlock], msg.get("content", []))
                    parts = convert_content_blocks_to_parts(content)
                    return A2AMessage(
                        kind="message",
                        role=Role.user,
                        parts=parts,
                        message_id=message_id,
                    )
        else:
            parts = convert_content_blocks_to_parts(cast(list[ContentBlock], prompt))
            return A2AMessage(
                kind="message",
                role=Role.user,
                parts=parts,
                message_id=message_id,
            )

    raise ValueError(f"Unsupported input type: {type(prompt)}")


def _convert_image_to_file_part(image: ImageContent) -> Part | None:
    """Convert Strands ImageContent to A2A FilePart.

    Args:
        image: Strands image content with format and source.

    Returns:
        A2A Part containing FilePart, or None if conversion fails.
    """
    source = image.get("source", {})
    mime_type = IMAGE_FORMAT_TO_MIME.get(image.get("format", "png"), "image/png")

    # Handle inline bytes
    if "bytes" in source and source["bytes"]:
        raw_bytes = source["bytes"]
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type=mime_type)
        return Part(FilePart(file=file_with_bytes, kind="file"))

    # Handle S3 or other location-based references
    if "location" in source:
        location = source["location"]
        if location.get("type") == "s3" and "uri" in location:
            file_with_uri = FileWithUri(uri=location["uri"], mime_type=mime_type)
            return Part(FilePart(file=file_with_uri, kind="file"))

    return None


def _convert_document_to_file_part(document: DocumentContent) -> Part | None:
    """Convert Strands DocumentContent to A2A FilePart.

    Args:
        document: Strands document content with format, name, and source.

    Returns:
        A2A Part containing FilePart, or None if conversion fails.
    """
    source = document.get("source", {})
    doc_format = document.get("format", "txt")
    mime_type = DOCUMENT_FORMAT_TO_MIME.get(doc_format, "application/octet-stream")
    name = document.get("name")

    # Handle inline bytes
    if "bytes" in source and source["bytes"]:
        raw_bytes = source["bytes"]
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type=mime_type, name=name)
        return Part(FilePart(file=file_with_bytes, kind="file"))

    # Handle S3 or other location-based references
    if "location" in source:
        location = source["location"]
        if location.get("type") == "s3" and "uri" in location:
            file_with_uri = FileWithUri(uri=location["uri"], mime_type=mime_type, name=name)
            return Part(FilePart(file=file_with_uri, kind="file"))

    return None


def _convert_video_to_file_part(video: VideoContent) -> Part | None:
    """Convert Strands VideoContent to A2A FilePart.

    Args:
        video: Strands video content with format and source.

    Returns:
        A2A Part containing FilePart, or None if conversion fails.
    """
    source = video.get("source", {})
    video_format = video.get("format", "mp4")
    mime_type = VIDEO_FORMAT_TO_MIME.get(video_format, "video/mp4")

    # Handle inline bytes
    if "bytes" in source and source["bytes"]:
        raw_bytes = source["bytes"]
        b64_str = base64.standard_b64encode(raw_bytes).decode("utf-8")
        file_with_bytes = FileWithBytes(bytes=b64_str, mime_type=mime_type)
        return Part(FilePart(file=file_with_bytes, kind="file"))

    # Handle S3 or other location-based references
    if "location" in source:
        location = source["location"]
        if location.get("type") == "s3" and "uri" in location:
            file_with_uri = FileWithUri(uri=location["uri"], mime_type=mime_type)
            return Part(FilePart(file=file_with_uri, kind="file"))

    return None


def convert_content_blocks_to_parts(content_blocks: list[ContentBlock]) -> list[Part]:
    """Convert Strands ContentBlocks to A2A Parts.

    Supports conversion of text, image, document, and video content blocks.

    Args:
        content_blocks: List of Strands content blocks.

    Returns:
        List of A2A Part objects.
    """
    parts = []
    for block in content_blocks:
        if "text" in block:
            parts.append(Part(TextPart(kind="text", text=block["text"])))
        elif "image" in block:
            part = _convert_image_to_file_part(block["image"])
            if part:
                parts.append(part)
        elif "document" in block:
            part = _convert_document_to_file_part(block["document"])
            if part:
                parts.append(part)
        elif "video" in block:
            part = _convert_video_to_file_part(block["video"])
            if part:
                parts.append(part)
    return parts


def _convert_file_part_to_content_block(file_part: FilePart) -> ContentBlock | None:
    """Convert A2A FilePart to Strands ContentBlock.

    Determines the content type based on MIME type and converts accordingly.

    Args:
        file_part: A2A FilePart containing file data or URI.

    Returns:
        Strands ContentBlock (image, document, or video), or None if unsupported.
    """
    file_data = file_part.file
    mime_type = file_data.mime_type or "application/octet-stream"

    # Check if it's an image
    if mime_type in MIME_TO_IMAGE_FORMAT:
        return _convert_file_part_to_image(file_data, mime_type)

    # Check if it's a document
    if mime_type in MIME_TO_DOCUMENT_FORMAT:
        return _convert_file_part_to_document(file_data, mime_type)

    # Check if it's a video
    if mime_type in MIME_TO_VIDEO_FORMAT:
        return _convert_file_part_to_video(file_data, mime_type)

    # Handle generic image/* mime types
    if mime_type.startswith("image/"):
        return _convert_file_part_to_image(file_data, mime_type)

    # Handle generic video/* mime types
    if mime_type.startswith("video/"):
        return _convert_file_part_to_video(file_data, mime_type)

    # Handle generic application/* and text/* as documents
    if mime_type.startswith("application/") or mime_type.startswith("text/"):
        return _convert_file_part_to_document(file_data, mime_type)

    return None


def _convert_file_part_to_image(
    file_data: FileWithBytes | FileWithUri,
    mime_type: str,
) -> ContentBlock:
    """Convert A2A file data to Strands ImageContent block.

    Args:
        file_data: A2A file with bytes or URI.
        mime_type: MIME type of the image.

    Returns:
        Strands ContentBlock containing ImageContent.
    """
    image_format: ImageFormat = MIME_TO_IMAGE_FORMAT.get(mime_type, "png")

    if isinstance(file_data, FileWithBytes):
        raw_bytes = base64.standard_b64decode(file_data.bytes)
        image_content: ImageContent = {
            "format": image_format,
            "source": {"bytes": raw_bytes},
        }
    else:
        # FileWithUri - use S3 location format
        image_content = {
            "format": image_format,
            "source": {"location": {"type": "s3", "uri": file_data.uri}},
        }

    return cast(ContentBlock, {"image": image_content})


def _convert_file_part_to_document(
    file_data: FileWithBytes | FileWithUri,
    mime_type: str,
) -> ContentBlock:
    """Convert A2A file data to Strands DocumentContent block.

    Args:
        file_data: A2A file with bytes or URI.
        mime_type: MIME type of the document.

    Returns:
        Strands ContentBlock containing DocumentContent.
    """
    doc_format: DocumentFormat = MIME_TO_DOCUMENT_FORMAT.get(mime_type, "txt")
    name = file_data.name

    if isinstance(file_data, FileWithBytes):
        raw_bytes = base64.standard_b64decode(file_data.bytes)
        doc_content: DocumentContent = {
            "format": doc_format,
            "source": {"bytes": raw_bytes},
        }
        if name:
            doc_content["name"] = name
    else:
        # FileWithUri - use S3 location format
        doc_content = {
            "format": doc_format,
            "source": {"location": {"type": "s3", "uri": file_data.uri}},
        }
        if name:
            doc_content["name"] = name

    return cast(ContentBlock, {"document": doc_content})


def _convert_file_part_to_video(
    file_data: FileWithBytes | FileWithUri,
    mime_type: str,
) -> ContentBlock:
    """Convert A2A file data to Strands VideoContent block.

    Args:
        file_data: A2A file with bytes or URI.
        mime_type: MIME type of the video.

    Returns:
        Strands ContentBlock containing VideoContent.
    """
    video_format: VideoFormat = MIME_TO_VIDEO_FORMAT.get(mime_type, "mp4")

    if isinstance(file_data, FileWithBytes):
        raw_bytes = base64.standard_b64decode(file_data.bytes)
        video_content: VideoContent = {
            "format": video_format,
            "source": {"bytes": raw_bytes},
        }
    else:
        # FileWithUri - use S3 location format
        video_content = {
            "format": video_format,
            "source": {"location": {"type": "s3", "uri": file_data.uri}},
        }

    return cast(ContentBlock, {"video": video_content})


def _extract_content_from_parts(parts: list[Part]) -> list[ContentBlock]:
    """Extract Strands ContentBlocks from A2A Parts.

    Supports extraction of text, image, document, and video parts.

    Args:
        parts: List of A2A Part objects.

    Returns:
        List of Strands ContentBlock objects.
    """
    content: list[ContentBlock] = []
    for part in parts:
        if hasattr(part, "root"):
            root = part.root
            if hasattr(root, "text"):
                content.append({"text": root.text})
            elif hasattr(root, "file"):
                block = _convert_file_part_to_content_block(root)
                if block:
                    content.append(block)
    return content


def convert_response_to_agent_result(response: A2AResponse) -> AgentResult:
    """Convert A2A response to AgentResult.

    Supports conversion of text, image, document, and video content types.

    Args:
        response: A2A response (either A2AMessage or tuple of task and update event).

    Returns:
        AgentResult with extracted content and metadata.
    """
    content: list[ContentBlock] = []

    if isinstance(response, tuple) and len(response) == 2:
        task, update_event = response

        # Handle artifact updates
        if isinstance(update_event, TaskArtifactUpdateEvent):
            if update_event.artifact and hasattr(update_event.artifact, "parts"):
                content.extend(_extract_content_from_parts(update_event.artifact.parts))
        # Handle status updates with messages
        elif isinstance(update_event, TaskStatusUpdateEvent):
            if update_event.status and hasattr(update_event.status, "message") and update_event.status.message:
                content.extend(_extract_content_from_parts(update_event.status.message.parts))
        # Handle initial task or task without update event
        elif update_event is None and task and hasattr(task, "artifacts") and task.artifacts is not None:
            for artifact in task.artifacts:
                if hasattr(artifact, "parts"):
                    content.extend(_extract_content_from_parts(artifact.parts))
    elif isinstance(response, A2AMessage):
        content.extend(_extract_content_from_parts(response.parts))

    message: Message = {
        "role": "assistant",
        "content": content,
    }

    return AgentResult(
        stop_reason="end_turn",
        message=message,
        metrics=EventLoopMetrics(),
        state={},
    )
