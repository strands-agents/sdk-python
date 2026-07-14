"""In-memory :class:`~strands.storage.Storage` backend."""

from __future__ import annotations

from typing import Any

from .base import Storage, namespace, normalize_key, normalize_prefix


class InMemoryStorage:
    """In-memory :class:`~strands.storage.Storage` backend backed by a ``dict``.

    Useful for testing and for serverless environments where disk access is
    unavailable. Content does not survive process restarts — for persistence use
    :class:`~strands.storage.LocalFileStorage` or :class:`~strands.storage.S3Storage`.

    This is a plain unbounded store with no eviction. Consumers that need eviction
    manage it themselves.

    Keys are normalized identically to :class:`~strands.storage.LocalFileStorage`:
    slash runs are collapsed, leading/trailing slashes are stripped, and ``..``
    segments are rejected.

    Example:
        ```python
        storage = InMemoryStorage()
        await storage.write("memory/notes.json", b"[]")
        data = await storage.read("memory/notes.json")
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._store: dict[str, bytes] = {}

    async def write(self, key: str, data: bytes, **kwargs: Any) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.

        The bytes are copied on write so later mutation of a caller-supplied
        ``bytearray`` cannot alias the stored value.

        Args:
            key: Opaque, ``/``-separated key identifying the value.
            data: Raw bytes to persist.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            StorageError: If the key is empty or contains ``..`` segments.
        """
        self._store[normalize_key(key)] = bytes(data)

    async def read(self, key: str, **kwargs: Any) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: The key to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The stored bytes, or ``None`` if no value exists for ``key``. The
            returned ``bytes`` object is immutable, so it never aliases the store.

        Raises:
            StorageError: If the key is empty or contains ``..`` segments.
        """
        return self._store.get(normalize_key(key))

    async def delete(self, key: str, **kwargs: Any) -> None:
        """Delete the value stored under ``key``. A no-op if the key does not exist.

        Args:
            key: The key to delete.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            StorageError: If the key is empty or contains ``..`` segments.
        """
        self._store.pop(normalize_key(key), None)

    async def list(self, prefix: str, **kwargs: Any) -> list[str]:
        """List the keys whose names begin with ``prefix``, sorted lexicographically.

        Args:
            prefix: Key prefix to match. An empty string matches all keys.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The matching keys, sorted ascending.

        Raises:
            StorageError: If the prefix contains ``..`` segments.
        """
        normalized = normalize_prefix(prefix)
        return sorted(key for key in self._store if key.startswith(normalized))

    def namespace(self, prefix: str) -> Storage:
        """Return a prefixed view of this storage without mutating the original."""
        return namespace(self, prefix)

    def clear(self) -> None:
        """Remove all stored entries. Useful for resetting state between tests."""
        self._store.clear()
