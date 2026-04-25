"""Storage backends for offloaded tool result content.

This module defines the Storage protocol and provides three built-in
implementations: file-based, in-memory, and S3 storage. Each content block
from a tool result is stored individually with its content type preserved.

Example:
    ```python
    from strands.vended_plugins.context_offloader import (
        FileStorage,
        InMemoryStorage,
        S3Storage,
    )

    # File-based storage
    storage = FileStorage(artifact_dir="./artifacts")
    ref = storage.store("tool_123_0", b"large output content...", "text/plain")
    content, content_type = storage.retrieve(ref)

    # In-memory storage (useful for testing and serverless)
    storage = InMemoryStorage()

    # S3 storage
    storage = S3Storage(bucket="my-bucket", prefix="artifacts/")
    ```
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError


def _sanitize_id(raw_id: str) -> str:
    """Sanitize an ID for safe use in filenames and object keys.

    Replaces path separators, parent directory references, and other
    unsafe characters with underscores.

    Args:
        raw_id: The raw ID string.

    Returns:
        A sanitized string safe for use in filenames.
    """
    sanitized = raw_id.replace("..", "_").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
    return sanitized


@runtime_checkable
class Storage(Protocol):
    """Backend for storing and retrieving offloaded content blocks.

    Each content block from a tool result is stored individually with its
    content type preserved. The SDK ships three built-in implementations:
    ``InMemoryStorage``, ``FileStorage``, and ``S3Storage``. Implement this
    protocol to create custom storage backends (e.g., Redis, DynamoDB).

    Lifecycle:
        This protocol intentionally does not include eviction or deletion methods.
        Stored content accumulates for the lifetime of the storage instance. For
        long-running agents, create a new storage instance per session or use a
        backend with built-in lifecycle management (e.g., S3 lifecycle policies).
    """

    def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content and return a reference identifier.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content (e.g., "text/plain",
                "application/json", "image/png", "application/pdf").

        Returns:
            A reference string that can be used to retrieve the content later.
        """
        ...

    def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve stored content by reference.

        Args:
            reference: The reference returned by a previous store() call.

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the reference is not found.
        """
        ...


class FileStorage:
    """Store offloaded content as files on disk.

    Files are written to the configured artifact directory with unique names.
    File extensions are derived from the content type. A ``.metadata.json``
    sidecar file tracks content types so they survive process restarts.

    Args:
        artifact_dir: Directory path where artifact files will be stored.
    """

    _METADATA_FILE = ".metadata.json"

    def __init__(self, artifact_dir: str = "./artifacts") -> None:
        """Initialize file-based storage.

        Args:
            artifact_dir: Directory path where artifact files will be stored.
        """
        self._artifact_dir = Path(artifact_dir)
        self._counter: int = 0
        self._lock = threading.Lock()
        self._content_types: dict[str, str] = self._load_metadata()

    @staticmethod
    def _extension_for(content_type: str) -> str:
        """Return a file extension for the given content type."""
        if content_type == "text/plain":
            return ".txt"
        return f".{content_type.split('/')[-1]}"

    def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content as a file and return the filename as reference.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            The filename (not full path) used as the reference.
        """
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        sanitized_key = _sanitize_id(key)
        timestamp_ms = int(time.time() * 1000)
        ext = self._extension_for(content_type)
        with self._lock:
            self._counter += 1
            counter = self._counter
            filename = f"{timestamp_ms}_{counter}_{sanitized_key}{ext}"
            self._content_types[filename] = content_type
            self._save_metadata()

        file_path = self._artifact_dir / filename
        file_path.write_bytes(content)

        return filename

    def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from a stored file.

        Args:
            reference: The filename reference returned by store().

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the file does not exist.
        """
        file_path = (self._artifact_dir / reference).resolve()
        if not file_path.is_relative_to(self._artifact_dir.resolve()):
            raise KeyError(f"Reference not found: {reference}")
        if not file_path.is_file():
            raise KeyError(f"Reference not found: {reference}")
        content_type = self._content_types.get(reference, "application/octet-stream")
        return file_path.read_bytes(), content_type

    def _load_metadata(self) -> dict[str, str]:
        """Load content type metadata from the sidecar file."""
        metadata_path = self._artifact_dir / self._METADATA_FILE
        if metadata_path.is_file():
            try:
                result: dict[str, str] = json.loads(metadata_path.read_text(encoding="utf-8"))
                return result
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save content type metadata to the sidecar file."""
        metadata_path = self._artifact_dir / self._METADATA_FILE
        metadata_path.write_text(json.dumps(self._content_types), encoding="utf-8")


class InMemoryStorage:
    """Store offloaded content in memory.

    Useful for testing and serverless environments where disk access
    is not available or not desired. Thread-safe.

    Note:
        Content accumulates for the lifetime of this instance. For long-running
        agents, consider creating a new instance per session or switching to
        ``FileStorage`` or ``S3Storage`` for persistent storage with external
        lifecycle management.
    """

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._store: dict[str, tuple[bytes, str]] = {}
        self._counter: int = 0
        self._lock = threading.Lock()

    def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content in memory and return a reference.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            A unique reference string.
        """
        with self._lock:
            self._counter += 1
            reference = f"mem_{self._counter}_{key}"
            self._store[reference] = (content, content_type)
        return reference

    def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from memory.

        Args:
            reference: The reference returned by store().

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the reference is not found.
        """
        with self._lock:
            if reference not in self._store:
                raise KeyError(f"Reference not found: {reference}")
            return self._store[reference]

    def clear(self) -> None:
        """Remove all stored content.

        Call this to free memory when offloaded results are no longer needed,
        e.g., between sessions or after an invocation completes.
        """
        with self._lock:
            self._store.clear()


class S3Storage:
    """Store offloaded content in Amazon S3.

    Objects are stored with unique keys under the configured prefix.
    Content type is preserved as S3 object metadata.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix for organizing stored artifacts.
        boto_session: Optional boto3 session. If not provided, a new session
            is created using the given region_name.
        boto_client_config: Optional botocore client configuration.
        region_name: AWS region. Used only when boto_session is not provided.

    Example:
        ```python
        from strands.vended_plugins.context_offloader import S3Storage

        storage = S3Storage(
            bucket="my-agent-artifacts",
            prefix="tool-results/",
        )
        ```
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        region_name: str | None = None,
    ) -> None:
        """Initialize S3-based storage.

        Args:
            bucket: S3 bucket name.
            prefix: S3 key prefix for organizing stored artifacts.
            boto_session: Optional boto3 session. If not provided, a new session
                is created using the given region_name.
            boto_client_config: Optional botocore client configuration.
            region_name: AWS region. Used only when boto_session is not provided.
        """
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if self._prefix:
            self._prefix += "/"

        session = boto_session or boto3.Session(region_name=region_name)

        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)
            new_user_agent = f"{existing_user_agent} strands-agents" if existing_user_agent else "strands-agents"
            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")

        self._client: Any = session.client(service_name="s3", config=client_config)
        self._counter: int = 0
        self._lock = threading.Lock()

    def store(self, key: str, content: bytes, content_type: str = "text/plain") -> str:
        """Store content as an S3 object and return the object key as reference.

        Args:
            key: A unique key for this content block.
            content: The raw content bytes to store.
            content_type: MIME type of the content.

        Returns:
            The S3 object key used as the reference.

        Raises:
            botocore.exceptions.ClientError: If the S3 operation fails (e.g., bucket
                does not exist, permission denied).
        """
        sanitized_key = _sanitize_id(key)
        timestamp_ms = int(time.time() * 1000)
        with self._lock:
            self._counter += 1
            counter = self._counter
        s3_key = f"{self._prefix}{timestamp_ms}_{counter}_{sanitized_key}"

        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=content,
            ContentType=content_type,
        )

        return s3_key

    def retrieve(self, reference: str) -> tuple[bytes, str]:
        """Retrieve content from an S3 object.

        Args:
            reference: The S3 object key returned by store().

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            KeyError: If the object does not exist.
        """
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=reference)
            content: bytes = response["Body"].read()
            content_type: str = response.get("ContentType", "application/octet-stream")
            return content, content_type
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"Reference not found: {reference}") from e
            raise
