"""Media-related type definitions for the SDK.

These types are modeled after the Bedrock API.

- Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Types_Amazon_Bedrock_Runtime.html
"""

from typing import Literal

from typing_extensions import TypedDict

DocumentFormat = Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
"""Supported document formats."""


class S3Location(TypedDict, total=False):
    """Contains the S3 location information for a document.

    Attributes:
        bucket: The S3 bucket name.
        key: The S3 object key.

    Note:
        Both bucket and key are required for a valid S3 location,
        but they are marked as optional in the type definition to allow
        for runtime validation in the code.
    """

    bucket: str
    key: str


class DocumentSource(TypedDict, total=False):
    """Contains the content of a document.

    Attributes:
        bytes: The binary content of the document.
        s3Location: The S3 location of the document (for Bedrock Nova models).
    """

    bytes: bytes
    s3Location: S3Location


class DocumentContent(TypedDict):
    """A document to include in a message.

    Attributes:
        format: The format of the document (e.g., "pdf", "txt").
        name: The name of the document.
        source: The source containing the document's binary content or S3 location.
    """

    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: DocumentSource


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
