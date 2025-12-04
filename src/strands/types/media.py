"""Media-related type definitions for the SDK.

These types are modeled after the Bedrock API.

- Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Types_Amazon_Bedrock_Runtime.html
"""

from typing import Any, List, Literal, Optional

from typing_extensions import TypedDict

from .citations import CitationsConfig

DocumentFormat = Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
"""Supported document formats."""


class DocumentSource(TypedDict, total=False):
    """Contains the content of a document.

    Attributes:
        bytes: The binary content of the document.
        text: The text content of the document source.
        content: The structured content of the document source.
    """

    bytes: bytes
    text: str
    content: List[Any]


class DocumentContent(TypedDict, total=False):
    """A document to include in a message.

    Attributes:
        format: The format of the document (e.g., "pdf", "txt").
        name: The name of the document.
        source: The source containing the document's binary content.
    """

    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: DocumentSource
    citations: Optional[CitationsConfig]
    context: Optional[str]


ImageFormat = Literal["png", "jpeg", "gif", "webp"]
"""Supported image formats."""


class ImageSource(TypedDict):
    """Contains the content of an image.

    Attributes:
        bytes: The binary content of the image.
    """

    bytes: bytes


class ImageContent(TypedDict):
    """An image to include in a message.

    Attributes:
        format: The format of the image (e.g., "png", "jpeg").
        source: The source containing the image's binary content.
    """

    format: ImageFormat
    source: ImageSource


VideoFormat = Literal["flv", "mkv", "mov", "mpeg", "mpg", "mp4", "three_gp", "webm", "wmv"]
"""Supported video formats."""


class VideoSource(TypedDict):
    """Contains the content of a video.

    Attributes:
        bytes: The binary content of the video.
    """

    bytes: bytes


class VideoContent(TypedDict):
    """A video to include in a message.

    Attributes:
        format: The format of the video (e.g., "mp4", "avi").
        source: The source containing the video's binary content.
    """

    format: VideoFormat
    source: VideoSource
