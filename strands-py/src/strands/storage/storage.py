"""Unified storage interface and key-normalization helpers."""

from __future__ import annotations

import builtins
import re
from typing import Protocol, runtime_checkable

from typing_extensions import TypeVar

from ..types.exceptions import StorageError

ListQuery = TypeVar("ListQuery", default=str, contravariant=True)

_NAMESPACED: object = object()
"""Internal sentinel marking a storage view as already namespace-scoped.

SDK constructs use this to detect whether the caller already scoped the storage,
so the default auto-prefix can be skipped.
"""


def _normalize_key(key: str) -> str:
    """Validate and normalize a storage key.

    Collapses runs of '/', strips leading and trailing '/', rejects empty
    keys, and rejects any '..' segment.

    Args:
        key: The raw key to normalize.

    Returns:
        The normalized key.

    Raises:
        StorageError: If the key is empty or contains a '..' segment.
    """
    normalized = re.sub(r"/+", "/", key).strip("/")
    if not normalized:
        raise StorageError("Storage key must not be empty")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage key '{key}': '..' path segments are not allowed")
    return normalized


def _normalize_prefix(prefix: str) -> str:
    """Normalize a list prefix.

    Collapses slash runs, strips leading slashes. Unlike a key, an empty
    prefix is valid and matches everything. A trailing slash is preserved
    because it is semantically significant for prefix matching.

    Args:
        prefix: The raw prefix to normalize.

    Returns:
        The normalized prefix.

    Raises:
        StorageError: If the prefix contains a '..' segment.
    """
    normalized = re.sub(r"/+", "/", prefix).lstrip("/")
    if ".." in normalized.split("/"):
        raise StorageError(f"Invalid storage prefix '{prefix}': '..' path segments are not allowed")
    return normalized


@runtime_checkable
class Storage(Protocol[ListQuery]):
    """A backend for storing and retrieving raw bytes under string keys.

    The interface is deliberately minimal — four operations over opaque bytes
    values. Implementations must treat keys as opaque path-like strings (segments
    separated by '/') and must round-trip the bytes they are given unchanged.

    The ``ListQuery`` type parameter controls what ``list`` accepts. It defaults to
    ``str`` (a key prefix), which every backend supports. Implementations may
    widen it to accept a richer query object while still accepting a plain string
    for SDK-internal callers.

    Implement this to add a custom backend; the SDK ships :class:`InMemoryStorage`,
    :class:`LocalFileStorage`, and :class:`S3Storage`.
    """

    async def write(self, key: str, data: bytes) -> None:
        """Store data under key, overwriting any existing value.

        Args:
            key: Opaque, '/'-separated key identifying the value.
            data: Raw bytes to persist.

        Raises:
            StorageError: If the write fails.
        """
        ...

    async def read(self, key: str) -> bytes | None:
        """Retrieve the bytes previously stored under key.

        Args:
            key: The key to read.

        Returns:
            The stored bytes, or None if no value exists for key.

        Raises:
            StorageError: If the read fails for a reason other than a missing key.
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete the value stored under key. A no-op if the key does not exist.

        Args:
            key: The key to delete.

        Raises:
            StorageError: If the delete fails.
        """
        ...

    async def list(self, query: ListQuery) -> builtins.list[str]:
        """List keys matching the given prefix query.

        Returns full keys (not the suffix after the prefix), sorted
        lexicographically. An empty string lists every key.

        Args:
            query: A string prefix to match.

        Returns:
            The matching keys, sorted ascending.

        Raises:
            StorageError: If the listing fails.
        """
        ...


class _NamespacedStorage:
    """A storage view that prepends a prefix to all keys.

    Composable — calling ``.namespace()`` on the result nests prefixes.
    """

    _namespaced = _NAMESPACED

    def __init__(self, storage: Storage, prefix: str) -> None:
        normalized = _normalize_prefix(prefix).rstrip("/")
        self._storage = storage
        self._prefix = f"{normalized}/" if normalized else ""

    async def write(self, key: str, data: bytes) -> None:
        """Store data under the prefixed key."""
        await self._storage.write(f"{self._prefix}{key}", data)

    async def read(self, key: str) -> bytes | None:
        """Read from the prefixed key."""
        return await self._storage.read(f"{self._prefix}{key}")

    async def delete(self, key: str) -> None:
        """Delete the prefixed key."""
        await self._storage.delete(f"{self._prefix}{key}")

    async def list(self, query: str = "") -> builtins.list[str]:
        """List keys under the prefix, stripping it from results."""
        keys = await self._storage.list(f"{self._prefix}{query}")
        return [key[len(self._prefix) :] for key in keys]

    def namespace(self, prefix: str) -> _NamespacedStorage:
        """Return a further-scoped view by nesting prefixes."""
        return _NamespacedStorage(self._storage, f"{self._prefix}{prefix}")

    def for_sandbox(self, sandbox: object) -> _NamespacedStorage:
        """Delegate sandbox binding to the underlying storage and re-wrap."""
        inner = self._storage
        if not hasattr(inner, "for_sandbox"):
            return self
        bound = inner.for_sandbox(sandbox)
        return _NamespacedStorage(bound, self._prefix.rstrip("/"))
