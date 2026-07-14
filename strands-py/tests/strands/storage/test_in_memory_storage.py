"""Tests for the in-memory unified storage backend."""

import pytest

from strands.storage import InMemoryStorage
from strands.types.exceptions import StorageError


@pytest.fixture
def storage():
    return InMemoryStorage()


class TestWrite:
    @pytest.mark.asyncio
    async def test_stores_data_under_the_given_key(self, storage):
        await storage.write("test/key", b"hello")
        assert await storage.read("test/key") == b"hello"

    @pytest.mark.asyncio
    async def test_overwrites_existing_data(self, storage):
        await storage.write("key", b"first")
        await storage.write("key", b"second")
        assert await storage.read("key") == b"second"

    @pytest.mark.asyncio
    async def test_copies_bytes_on_write_to_prevent_aliasing(self, storage):
        data = bytearray([1, 2, 3])
        await storage.write("key", data)
        data[0] = 99
        tru_result = await storage.read("key")
        assert tru_result[0] == 1


class TestRead:
    @pytest.mark.asyncio
    async def test_returns_none_for_missing_keys(self, storage):
        assert await storage.read("nonexistent") is None

    @pytest.mark.asyncio
    async def test_returned_bytes_are_immutable_and_do_not_alias(self, storage):
        await storage.write("key", bytes([1, 2, 3]))
        first = await storage.read("key")
        with pytest.raises(TypeError):
            first[0] = 99  # bytes is immutable — cannot alias the store
        assert (await storage.read("key"))[0] == 1


class TestDelete:
    @pytest.mark.asyncio
    async def test_removes_an_existing_key(self, storage):
        await storage.write("key", bytes([1]))
        await storage.delete("key")
        assert await storage.read("key") is None

    @pytest.mark.asyncio
    async def test_is_a_no_op_for_missing_keys(self, storage):
        assert await storage.delete("nonexistent") is None


class TestList:
    @pytest.mark.asyncio
    async def test_returns_keys_matching_a_prefix(self, storage):
        await storage.write("sessions/a/data", bytes([1]))
        await storage.write("sessions/b/data", bytes([2]))
        await storage.write("memory/notes", bytes([3]))

        tru_keys = await storage.list("sessions/")
        assert tru_keys == ["sessions/a/data", "sessions/b/data"]

    @pytest.mark.asyncio
    async def test_returns_all_keys_when_prefix_is_empty(self, storage):
        await storage.write("a", bytes([1]))
        await storage.write("b", bytes([2]))
        assert await storage.list("") == ["a", "b"]

    @pytest.mark.asyncio
    async def test_returns_keys_sorted_lexicographically(self, storage):
        await storage.write("c", bytes([3]))
        await storage.write("a", bytes([1]))
        await storage.write("b", bytes([2]))
        assert await storage.list("") == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_keys_match(self, storage):
        await storage.write("other/key", bytes([1]))
        assert await storage.list("sessions/") == []


class TestClear:
    @pytest.mark.asyncio
    async def test_removes_all_entries(self, storage):
        await storage.write("a", bytes([1]))
        await storage.write("b", bytes([2]))
        storage.clear()
        assert await storage.list("") == []


class TestKeyNormalization:
    @pytest.mark.asyncio
    async def test_normalizes_slashes_so_equivalent_keys_resolve_to_the_same_entry(self, storage):
        await storage.write("/a//b/", bytes([1]))
        assert await storage.read("a/b") == bytes([1])

    @pytest.mark.asyncio
    async def test_rejects_empty_keys(self, storage):
        with pytest.raises(StorageError):
            await storage.write("", bytes([1]))

    @pytest.mark.asyncio
    async def test_rejects_keys_with_dotdot_segments(self, storage):
        with pytest.raises(StorageError):
            await storage.write("a/../b", bytes([1]))

    @pytest.mark.asyncio
    async def test_rejects_prefixes_with_dotdot_segments(self, storage):
        with pytest.raises(StorageError):
            await storage.list("../")


class TestPublicApiSurface:
    def test_namespace_and_error_are_exported_from_package(self):
        from strands.storage import InMemoryStorage, S3Storage, Storage, namespace
        from strands.storage import LocalFileStorage as _LFS
        from strands.storage import StorageError as _SE

        assert callable(namespace)
        assert isinstance(InMemoryStorage(), Storage)
        assert issubclass(_SE, Exception)
        assert _LFS is not None and S3Storage is not None

    @pytest.mark.asyncio
    async def test_operations_tolerate_forward_compat_kwargs(self, storage):
        # Protocol methods carry **kwargs for forward compatibility; unknown kwargs
        # must not break existing implementations.
        await storage.write("k", b"v", request_id="abc")
        assert await storage.read("k", request_id="abc") == b"v"
        assert await storage.list("", request_id="abc") == ["k"]
        await storage.delete("k", request_id="abc")
        assert await storage.read("k") is None
