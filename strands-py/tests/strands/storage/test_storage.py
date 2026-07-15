"""Tests for the storage Protocol, normalize helpers, and namespace function."""

import pytest

from strands.storage.storage import (
    _NAMESPACED,
    _NamespacedStorage,
    _normalize_key,
    _normalize_prefix,
)
from strands.types.exceptions import StorageError


class TestNormalizeKey:
    def test_simple_key(self):
        assert _normalize_key("foo/bar") == "foo/bar"

    def test_collapses_multiple_slashes(self):
        assert _normalize_key("foo//bar///baz") == "foo/bar/baz"

    def test_strips_leading_slash(self):
        assert _normalize_key("/foo/bar") == "foo/bar"

    def test_strips_trailing_slash(self):
        assert _normalize_key("foo/bar/") == "foo/bar"

    def test_strips_both(self):
        assert _normalize_key("///foo///bar///") == "foo/bar"

    def test_rejects_empty_key(self):
        with pytest.raises(StorageError, match="must not be empty"):
            _normalize_key("")

    def test_rejects_only_slashes(self):
        with pytest.raises(StorageError, match="must not be empty"):
            _normalize_key("///")

    def test_rejects_dot_dot_segment(self):
        with pytest.raises(StorageError, match="path segments are not allowed"):
            _normalize_key("foo/../bar")

    def test_rejects_dot_dot_at_start(self):
        with pytest.raises(StorageError, match="path segments are not allowed"):
            _normalize_key("../foo")

    def test_allows_dots_in_segment(self):
        assert _normalize_key("foo/bar.txt") == "foo/bar.txt"
        assert _normalize_key("foo/...bar") == "foo/...bar"


class TestNormalizePrefix:
    def test_simple_prefix(self):
        assert _normalize_prefix("foo/bar") == "foo/bar"

    def test_empty_prefix_is_valid(self):
        assert _normalize_prefix("") == ""

    def test_collapses_slashes(self):
        assert _normalize_prefix("foo//bar///") == "foo/bar/"

    def test_strips_leading_slash(self):
        assert _normalize_prefix("/foo/bar") == "foo/bar"

    def test_preserves_trailing_slash(self):
        assert _normalize_prefix("foo/bar/") == "foo/bar/"

    def test_rejects_dot_dot(self):
        with pytest.raises(StorageError, match="path segments are not allowed"):
            _normalize_prefix("foo/../bar")


class TestNamespacedStorage:
    @pytest.fixture
    def storage(self):
        from strands.storage import InMemoryStorage

        return InMemoryStorage()

    def test_has_namespaced_sentinel(self, storage):
        ns = _NamespacedStorage(storage, "prefix")
        assert ns._namespaced is _NAMESPACED

    @pytest.mark.asyncio
    async def test_write_read_round_trip(self, storage):
        ns = _NamespacedStorage(storage, "scope")
        await ns.write("key", b"hello")
        assert await ns.read("key") == b"hello"
        # The underlying storage has the prefixed key
        assert await storage.read("scope/key") == b"hello"

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        ns = _NamespacedStorage(storage, "scope")
        await ns.write("key", b"data")
        await ns.delete("key")
        assert await ns.read("key") is None

    @pytest.mark.asyncio
    async def test_list_strips_prefix(self, storage):
        ns = _NamespacedStorage(storage, "scope")
        await ns.write("a", b"1")
        await ns.write("b", b"2")
        await storage.write("other/x", b"3")
        keys = await ns.list("")
        assert keys == ["a", "b"]

    @pytest.mark.asyncio
    async def test_composable_nesting(self, storage):
        ns1 = _NamespacedStorage(storage, "a")
        ns2 = ns1.namespace("b")
        await ns2.write("key", b"nested")
        assert await storage.read("a/b/key") == b"nested"
        assert await ns2.read("key") == b"nested"

    @pytest.mark.asyncio
    async def test_empty_prefix(self, storage):
        ns = _NamespacedStorage(storage, "")
        await ns.write("key", b"data")
        assert await storage.read("key") == b"data"

    @pytest.mark.asyncio
    async def test_trailing_slash_prefix_does_not_corrupt_keys(self, storage):
        ns = _NamespacedStorage(storage, "sessions/")
        await ns.write("key1", b"hello")
        keys = await ns.list("")
        assert keys == ["key1"]


class TestStorageProtocol:
    def test_isinstance_check(self):
        from strands.storage import InMemoryStorage, Storage

        storage = InMemoryStorage()
        assert isinstance(storage, Storage)
