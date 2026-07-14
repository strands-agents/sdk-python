"""Amazon S3 :class:`~strands.storage.Storage` backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError

from ..types.exceptions import StorageError
from .base import Storage, namespace, normalize_key, normalize_prefix

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

_S3_PAGE_SIZE = 1000
_NOT_FOUND_CODES = frozenset({"NoSuchKey", "NotFound", "404"})


class S3Storage:
    """Amazon S3 :class:`~strands.storage.Storage` backend.

    Stores each key as an S3 object under an optional prefix. The boto3 client is
    created lazily on first use, so constructing an ``S3Storage`` never requires
    AWS credentials or a configured region.

    Example:
        ```python
        from strands.storage import S3Storage

        storage = S3Storage("my-bucket", prefix="agents/")
        await storage.write("sessions/abc/snapshot.json", data)
        ```
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region: str | None = None,
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
    ) -> None:
        """Initialize S3-based storage.

        Args:
            bucket: Target S3 bucket name.
            prefix: Optional key prefix prepended to every key (a leading namespace
                within the bucket).
            region: AWS region override. When omitted, the standard boto3
                resolution chain applies. Cannot be combined with ``boto_session``.
            boto_session: Pre-configured boto3 session. Cannot be combined with
                ``region``.
            boto_client_config: Optional botocore client configuration.

        Raises:
            StorageError: If both ``region`` and ``boto_session`` are provided.
        """
        if boto_session is not None and region is not None:
            raise StorageError(
                "Cannot specify both boto_session and region. Configure the region on the boto session instead."
            )
        self._bucket = bucket
        self._prefix = f"{prefix.rstrip('/')}/" if prefix else ""
        self._region = region
        self._boto_session = boto_session
        self._boto_client_config = boto_client_config
        self._client: S3Client | None = None

    async def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.

        Args:
            key: Opaque, ``/``-separated key identifying the value.
            data: Raw bytes to persist.

        Raises:
            StorageError: If the key is invalid or the upload fails.
        """
        normalized = normalize_key(key)
        client = self._get_client()
        try:
            client.put_object(Bucket=self._bucket, Key=self._object_key(normalized), Body=bytes(data))
        except Exception as error:
            raise StorageError(f"Failed to write '{normalized}' to S3 bucket '{self._bucket}'") from error

    async def read(self, key: str) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: The key to read.

        Returns:
            The stored bytes, or ``None`` if no value exists for ``key``.

        Raises:
            StorageError: If the key is invalid or the download fails.
        """
        normalized = normalize_key(key)
        client = self._get_client()
        try:
            response = client.get_object(Bucket=self._bucket, Key=self._object_key(normalized))
            body: bytes = response["Body"].read()
            return body
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in _NOT_FOUND_CODES:
                return None
            raise StorageError(f"Failed to read '{normalized}' from S3 bucket '{self._bucket}'") from error
        except Exception as error:
            raise StorageError(f"Failed to read '{normalized}' from S3 bucket '{self._bucket}'") from error

    async def delete(self, key: str) -> None:
        """Delete the value stored under ``key``. A no-op if the key does not exist.

        Args:
            key: The key to delete.

        Raises:
            StorageError: If the key is invalid or the delete request fails.
        """
        normalized = normalize_key(key)
        client = self._get_client()
        try:
            client.delete_object(Bucket=self._bucket, Key=self._object_key(normalized))
        except Exception as error:
            raise StorageError(f"Failed to delete '{normalized}' from S3 bucket '{self._bucket}'") from error

    async def list(self, prefix: str) -> list[str]:
        """List the keys whose names begin with ``prefix``, sorted lexicographically.

        Args:
            prefix: Key prefix to match. An empty string matches all keys.

        Returns:
            The matching keys, sorted ascending.

        Raises:
            StorageError: If the prefix is invalid or the list request fails.
        """
        normalized = normalize_prefix(prefix)
        client = self._get_client()
        list_prefix = f"{self._prefix}{normalized}"
        keys: list[str] = []
        continuation_token: str | None = None
        try:
            while True:
                params: dict[str, Any] = {
                    "Bucket": self._bucket,
                    "Prefix": list_prefix,
                    "MaxKeys": _S3_PAGE_SIZE,
                }
                if continuation_token is not None:
                    params["ContinuationToken"] = continuation_token
                response = client.list_objects_v2(**params)
                for obj in response.get("Contents", []):
                    obj_key = obj.get("Key")
                    if obj_key is None:
                        continue
                    keys.append(obj_key[len(self._prefix) :] if self._prefix else obj_key)
                # Continue only while the response is truncated AND hands back a token;
                # a truncated response with no token terminates (matches strands-ts).
                continuation_token = response.get("NextContinuationToken") if response.get("IsTruncated") else None
                if not continuation_token:
                    break
        except Exception as error:
            raise StorageError(f"Failed to list S3 bucket '{self._bucket}' under '{normalized}'") from error
        return sorted(keys)

    def _get_client(self) -> S3Client:
        if self._client is not None:
            return self._client
        session = self._boto_session or boto3.Session(region_name=self._region)
        if self._boto_client_config:
            existing_user_agent = getattr(self._boto_client_config, "user_agent_extra", None)
            new_user_agent = f"{existing_user_agent} strands-agents" if existing_user_agent else "strands-agents"
            client_config = self._boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents")
        self._client = session.client("s3", config=client_config)
        return self._client

    def namespace(self, prefix: str) -> Storage:
        """Return a prefixed view of this storage without mutating the original."""
        return namespace(self, prefix)

    def _object_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
