"""Tests for LocalFileStorage."""

import os

import pytest

from strands.storage import LocalFileStorage
from strands.types.exceptions import StorageError


class TestLocalFileStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        return LocalFileStorage(str(tmp_path) + "/")

    @pytest.mark.asyncio
    async def test_write_and_read(self, storage):
        await storage.write("key.txt", b"hello")
        assert await storage.read("key.txt") == b"hello"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, storage, tmp_path):
        await storage.write("a/b/c/file.txt", b"deep")
        assert await storage.read("a/b/c/file.txt") == b"deep"
        assert os.path.isfile(os.path.join(str(tmp_path), "a/b/c/file.txt"))

    @pytest.mark.asyncio
    async def test_read_missing_returns_none(self, storage):
        assert await storage.read("nonexistent") is None

    @pytest.mark.asyncio
    async def test_write_overwrites(self, storage):
        await storage.write("key", b"first")
        await storage.write("key", b"second")
        assert await storage.read("key") == b"second"

    @pytest.mark.asyncio
    async def test_delete_existing(self, storage):
        await storage.write("key", b"data")
        await storage.delete("key")
        assert await storage.read("key") is None

    @pytest.mark.asyncio
    async def test_delete_missing_is_noop(self, storage):
        await storage.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_list_all(self, storage):
        await storage.write("b", b"")
        await storage.write("a", b"")
        await storage.write("c", b"")
        keys = await storage.list("")
        assert keys == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_with_prefix(self, storage):
        await storage.write("sessions/a", b"")
        await storage.write("sessions/b", b"")
        await storage.write("offloader/x", b"")
        keys = await storage.list("sessions/")
        assert keys == ["sessions/a", "sessions/b"]

    @pytest.mark.asyncio
    async def test_list_nested(self, storage):
        await storage.write("a/1", b"")
        await storage.write("a/2", b"")
        await storage.write("a/sub/3", b"")
        keys = await storage.list("a/")
        assert keys == ["a/1", "a/2", "a/sub/3"]

    @pytest.mark.asyncio
    async def test_list_empty_dir(self, storage):
        assert await storage.list("") == []

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, storage):
        with pytest.raises(StorageError):
            await storage.write("../escape", b"data")

    @pytest.mark.asyncio
    async def test_rejects_empty_key(self, storage):
        with pytest.raises(StorageError):
            await storage.write("", b"data")

    @pytest.mark.asyncio
    async def test_atomic_write(self, storage, tmp_path):
        await storage.write("file.txt", b"content")
        # No temp files remain
        all_files = list(tmp_path.rglob("*"))
        assert not any("__strands_tmp" in str(f) for f in all_files)

    @pytest.mark.asyncio
    async def test_namespace(self, storage):
        ns = storage.namespace("scope")
        await ns.write("key", b"value")
        assert await ns.read("key") == b"value"
        assert await storage.read("scope/key") == b"value"

    @pytest.mark.asyncio
    async def test_key_normalization(self, storage):
        await storage.write("//foo///bar//", b"data")
        assert await storage.read("foo/bar") == b"data"

    def test_for_sandbox_returns_self_if_same(self, tmp_path):
        from unittest.mock import MagicMock

        sandbox = MagicMock()
        storage = LocalFileStorage(str(tmp_path), sandbox=sandbox)
        assert storage.for_sandbox(sandbox) is storage

    def test_for_sandbox_returns_new_if_different(self, tmp_path):
        from unittest.mock import MagicMock

        sandbox1 = MagicMock()
        sandbox2 = MagicMock()
        storage = LocalFileStorage(str(tmp_path), sandbox=sandbox1)
        new_storage = storage.for_sandbox(sandbox2)
        assert new_storage is not storage
