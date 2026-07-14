"""Core contract and helpers for the unified storage primitive.

This module defines the :class:`Storage` protocol — the SDK's single persistence
primitive — plus internal helpers for key normalization and namespacing. The
shipped backends live in sibling modules and are re-exported from the package
root.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from ..types.exceptions import StorageError

# Attribute name marking a namespaced storage view. Constructs check for it (via
# ``getattr``) to detect whether the caller already scoped the storage, so they
# can skip applying a default prefix. This is the Python analog of the TypeScript
# ``NAMESPACED`` symbol.
_NAMESPACED = "_strands_storage_namespaced"

_SLASH_RUN = re.compile(r"/+")


def normalize_key(key: str) -> str:
    """Validate and normalize a storage key.

    Collapses runs of ``/``, strips leading and trailing ``/``, and rejects empty
    keys and any ``..`` segment.

    Args:
        key: The raw key to normalize.

    Returns:
        The normalized key.

    Raises:
        StorageError: If the key is empty or contains a ``..`` segment.
    """
    normalized = _SLASH_RUN.sub("/", key).strip("/")
    if not normalized:
        raise StorageError("Storage key must not be empty")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage key '{key}': '..' path segments are not allowed")
    return normalized


def normalize_prefix(prefix: str) -> str:
    """Normalize a list prefix.

    Collapses slash runs and strips leading slashes. Unlike a key, an empty prefix
    is valid and matches everything, and a trailing slash is preserved.

    Args:
        prefix: The raw prefix to normalize.

    Returns:
        The normalized prefix.

    Raises:
        StorageError: If the prefix contains a ``..`` segment.
    """
    normalized = _SLASH_RUN.sub("/", prefix).lstrip("/")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage prefix '{prefix}': '..' path segments are not allowed")
    return normalized


@runtime_checkable
class Storage(Protocol):
    """A backend for storing and retrieving raw bytes under string keys.

    The interface is deliberately minimal — four operations over opaque ``bytes``
    values. Implementations must treat keys as opaque path-like strings (segments
    separated by ``/``) and must round-trip the bytes they are given unchanged.

    By default ``list`` performs a prefix match over a string, which every backend
    supports. An implementation may *widen* the accepted ``prefix`` parameter to a
    richer query object (e.g. a DynamoDB partition/sort-key filter) while still
    accepting a plain string for SDK-internal callers.

    Implement this to add a custom backend; the SDK ships :class:`InMemoryStorage`,
    :class:`LocalFileStorage`, and :class:`S3Storage`. Those shipped backends also
    expose a ``namespace(prefix)`` convenience method; a custom backend can instead
    be scoped with the standalone :func:`namespace` factory (exported from
    ``strands.storage``).

    Example:
        ```python
        from strands.storage import InMemoryStorage

        storage = InMemoryStorage()
        await storage.write("memory/notes.json", b"[]")
        data = await storage.read("memory/notes.json")
        ```
    """

    async def write(self, key: str, data: bytes, **kwargs: Any) -> None:
        """Store ``data`` under ``key``, overwriting any existing value.

        Args:
            key: Opaque, ``/``-separated key identifying the value.
            data: Raw bytes to persist.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            StorageError: If the key is invalid or the write fails.
        """
        ...

    async def read(self, key: str, **kwargs: Any) -> bytes | None:
        """Retrieve the bytes previously stored under ``key``.

        Args:
            key: The key to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The stored bytes, or ``None`` if no value exists for ``key``.

        Raises:
            StorageError: If the read fails for a reason other than a missing key.
        """
        ...

    async def delete(self, key: str, **kwargs: Any) -> None:
        """Delete the value stored under ``key``. A no-op if the key does not exist.

        Args:
            key: The key to delete.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            StorageError: If the delete fails.
        """
        ...

    async def list(self, prefix: str, **kwargs: Any) -> list[str]:
        """List keys matching the given prefix.

        Returns full keys (not the suffix after the prefix), sorted
        lexicographically. An empty string lists every key.

        Args:
            prefix: A key prefix. An empty string matches all keys.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The matching keys, sorted ascending.

        Raises:
            StorageError: If the listing fails.
        """
        ...


class _NamespacedStorage:
    """A :class:`Storage` view that prefixes every key with a fixed namespace.

    Returned by :func:`namespace`. Delegates to an underlying storage, prepending
    the prefix on write/read/delete/list and stripping it back off listed keys.
    """

    def __init__(self, storage: Storage, prefix: str) -> None:
        """Initialize the view.

        Args:
            storage: The underlying storage to delegate to.
            prefix: The already-normalized prefix, including a trailing ``/`` when
                non-empty (as produced by :func:`namespace`).
        """
        self._storage = storage
        self._prefix = prefix
        setattr(self, _NAMESPACED, True)

    async def write(self, key: str, data: bytes, **kwargs: Any) -> None:
        """Store ``data`` under the namespaced key."""
        await self._storage.write(f"{self._prefix}{key}", data, **kwargs)

    async def read(self, key: str, **kwargs: Any) -> bytes | None:
        """Read the bytes stored under the namespaced key."""
        return await self._storage.read(f"{self._prefix}{key}", **kwargs)

    async def delete(self, key: str, **kwargs: Any) -> None:
        """Delete the value stored under the namespaced key."""
        await self._storage.delete(f"{self._prefix}{key}", **kwargs)

    async def list(self, prefix: str, **kwargs: Any) -> list[str]:
        """List keys under the namespace, with the namespace prefix stripped off."""
        keys = await self._storage.list(f"{self._prefix}{prefix}", **kwargs)
        offset = len(self._prefix)
        return [key[offset:] for key in keys]

    def namespace(self, prefix: str) -> Storage:
        """Return a further-nested namespaced view of the underlying storage."""
        return namespace(self._storage, f"{self._prefix}{prefix}")


def namespace(storage: Storage, prefix: str) -> Storage:
    """Return a :class:`Storage` view with all keys prefixed by ``prefix``.

    The original storage is not mutated. Composable — calling ``namespace()`` on
    the result nests prefixes. An empty prefix yields a transparent pass-through.

    Args:
        storage: The underlying storage to delegate to.
        prefix: Prefix to prepend to all keys.

    Returns:
        A namespaced Storage view.
    """
    normalized = normalize_prefix(prefix)
    scoped = f"{normalized}/" if normalized else ""
    return _NamespacedStorage(storage, scoped)
