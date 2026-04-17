"""Storage backends for externalized tool results.

This module defines the ExternalizationStorage protocol and provides three
built-in implementations: file-based, in-memory, and S3 storage.

Example:
    ```python
    from strands.vended_plugins.tool_result_externalizer import (
        FileExternalizationStorage,
        InMemoryExternalizationStorage,
        S3ExternalizationStorage,
    )

    # File-based storage
    storage = FileExternalizationStorage(artifact_dir="./artifacts")
    ref = storage.store("tool_123", "large output content...")
    content = storage.retrieve(ref)

    # In-memory storage (useful for testing and serverless)
    storage = InMemoryExternalizationStorage()

    # S3 storage
    storage = S3ExternalizationStorage(bucket="my-bucket", prefix="artifacts/")
    ```
"""

import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError


def _sanitize_id(tool_use_id: str) -> str:
    """Sanitize a tool use ID for safe use in filenames and object keys.

    Replaces path separators, parent directory references, and other
    unsafe characters with underscores.

    Args:
        tool_use_id: The raw tool use ID.

    Returns:
        A sanitized string safe for use in filenames.
    """
    sanitized = tool_use_id.replace("..", "_").replace("/", "_").replace("\\", "_")
    sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
    return sanitized


@runtime_checkable
class ExternalizationStorage(Protocol):
    """Backend for storing and retrieving externalized tool results.

    The SDK ships three built-in implementations: ``InMemoryExternalizationStorage``,
    ``FileExternalizationStorage``, and ``S3ExternalizationStorage``. Implement this
    protocol to create custom storage backends (e.g., Redis, DynamoDB).
    """

    def store(self, tool_use_id: str, content: str) -> str:
        """Store content and return a reference identifier.

        Args:
            tool_use_id: The tool use ID that produced this content.
            content: The full text content to store.

        Returns:
            A reference string that can be used to retrieve the content later.
        """
        ...

    def retrieve(self, reference: str) -> str:
        """Retrieve stored content by reference.

        Args:
            reference: The reference returned by a previous store() call.

        Returns:
            The stored content.

        Raises:
            KeyError: If the reference is not found.
        """
        ...


class FileExternalizationStorage:
    """Store externalized tool results as files on disk.

    Files are written to the configured artifact directory with unique names
    based on timestamp, counter, and tool use ID.

    Args:
        artifact_dir: Directory path where artifact files will be stored.
    """

    def __init__(self, artifact_dir: str = "./artifacts") -> None:
        """Initialize file-based storage.

        Args:
            artifact_dir: Directory path where artifact files will be stored.
        """
        self._artifact_dir = Path(artifact_dir)
        self._counter: int = 0
        self._lock = threading.Lock()

    def store(self, tool_use_id: str, content: str) -> str:
        """Store content as a file and return the filename as reference.

        Args:
            tool_use_id: The tool use ID that produced this content.
            content: The full text content to store.

        Returns:
            The filename (not full path) used as the reference.
        """
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        sanitized_id = _sanitize_id(tool_use_id)
        timestamp_ms = int(time.time() * 1000)
        with self._lock:
            self._counter += 1
            counter = self._counter
        filename = f"{timestamp_ms}_{counter}_{sanitized_id}.txt"

        file_path = self._artifact_dir / filename
        file_path.write_text(content, encoding="utf-8")

        return filename

    def retrieve(self, reference: str) -> str:
        """Retrieve content from a stored file.

        Args:
            reference: The filename reference returned by store().

        Returns:
            The stored content.

        Raises:
            KeyError: If the file does not exist.
        """
        file_path = (self._artifact_dir / reference).resolve()
        if not file_path.is_relative_to(self._artifact_dir.resolve()):
            raise KeyError(f"Reference not found: {reference}")
        if not file_path.is_file():
            raise KeyError(f"Reference not found: {reference}")
        return file_path.read_text(encoding="utf-8")


class InMemoryExternalizationStorage:
    """Store externalized tool results in memory.

    Useful for testing and serverless environments where disk access
    is not available or not desired. Thread-safe.
    """

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._store: dict[str, str] = {}
        self._counter: int = 0
        self._lock = threading.Lock()

    def store(self, tool_use_id: str, content: str) -> str:
        """Store content in memory and return a reference.

        Args:
            tool_use_id: The tool use ID that produced this content.
            content: The full text content to store.

        Returns:
            A unique reference string.
        """
        with self._lock:
            self._counter += 1
            reference = f"mem_{self._counter}_{tool_use_id}"
            self._store[reference] = content
        return reference

    def retrieve(self, reference: str) -> str:
        """Retrieve content from memory.

        Args:
            reference: The reference returned by store().

        Returns:
            The stored content.

        Raises:
            KeyError: If the reference is not found.
        """
        with self._lock:
            if reference not in self._store:
                raise KeyError(f"Reference not found: {reference}")
            return self._store[reference]


class S3ExternalizationStorage:
    """Store externalized tool results in Amazon S3.

    Objects are stored as UTF-8 text with unique keys based on timestamp,
    counter, and tool use ID under the configured prefix.

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix for organizing stored artifacts.
        boto_session: Optional boto3 session. If not provided, a new session
            is created using the given region_name.
        boto_client_config: Optional botocore client configuration.
        region_name: AWS region. Used only when boto_session is not provided.

    Example:
        ```python
        from strands.vended_plugins.tool_result_externalizer import S3ExternalizationStorage

        storage = S3ExternalizationStorage(
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

    def store(self, tool_use_id: str, content: str) -> str:
        """Store content as an S3 object and return the object key as reference.

        Args:
            tool_use_id: The tool use ID that produced this content.
            content: The full text content to store.

        Returns:
            The S3 object key used as the reference.
        """
        sanitized_id = _sanitize_id(tool_use_id)
        timestamp_ms = int(time.time() * 1000)
        with self._lock:
            self._counter += 1
            counter = self._counter
        key = f"{self._prefix}{timestamp_ms}_{counter}_{sanitized_id}.txt"

        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )

        return key

    def retrieve(self, reference: str) -> str:
        """Retrieve content from an S3 object.

        Args:
            reference: The S3 object key returned by store().

        Returns:
            The stored content.

        Raises:
            KeyError: If the object does not exist.
        """
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=reference)
            body: str = response["Body"].read().decode("utf-8")
            return body
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"Reference not found: {reference}") from e
            raise
