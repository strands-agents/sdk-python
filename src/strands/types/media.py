"""Media-related type definitions for the SDK.

These types are modeled after the Bedrock API.

- Bedrock docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Types_Amazon_Bedrock_Runtime.html
"""

from typing import Literal

from typing_extensions import Required, TypedDict

from .citations import CitationsConfig

DocumentFormat = Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
"""Supported document formats."""


class S3Location(TypedDict, total=False):
    """A storage location in an Amazon S3 bucket.

    Used by Bedrock to reference media files stored in S3 instead of passing raw bytes.

    - Docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_S3Location.html

    Attributes:
        uri: An object URI starting with `s3://`. Required.
        bucketOwner: If the bucket belongs to another AWS account, specify that account's ID. Optional.
    """

    uri: Required[str]
    bucketOwner: str


class DocumentSource(TypedDict, total=False):
    """Contains the content of a document.

    Only one of `bytes` or `s3Location` should be specified.

    Attributes:
        bytes: The binary content of the document.
        s3Location: S3 location of the document (Bedrock only).
    """

    bytes: bytes
    s3Location: S3Location


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
    citations: CitationsConfig | None
    context: str | None


ImageFormat = Literal["png", "jpeg", "gif", "webp"]
"""Supported image formats."""


class ImageSource(TypedDict, total=False):
    """Contains the content of an image.

    Only one of `bytes` or `s3Location` should be specified.

    Attributes:
        bytes: The binary content of the image.
        s3Location: S3 location of the image (Bedrock only).
    """

    bytes: bytes
    s3Location: S3Location


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


class VideoSource(TypedDict, total=False):
    """Contains the content of a video.

    Only one of `bytes` or `s3Location` should be specified.

    Attributes:
        bytes: The binary content of the video.
        s3Location: S3 location of the video (Bedrock only).
    """

    bytes: bytes
    s3Location: S3Location


class VideoContent(TypedDict):
    """A video to include in a message.

    Attributes:
        format: The format of the video (e.g., "mp4", "avi").
        source: The source containing the video's binary content.
    """

    format: VideoFormat
    source: VideoSource
