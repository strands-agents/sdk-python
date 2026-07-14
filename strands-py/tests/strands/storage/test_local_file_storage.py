"""Tests for the local-filesystem unified storage backend."""

import os
import sys
from unittest.mock import patch

import pytest

from strands.storage import LocalFileStorage
from strands.types.exceptions import StorageError


@pytest.fixture
def base_dir(tmp_path):
    return str(tmp_path / "strands-test")


@pytest.fixture
def storage(base_dir):
    return LocalFileStorage(base_dir)


class TestWriteAndRead:
    @pytest.mark.asyncio
    async def test_round_trips_bytes(self, storage):
        data = b"hello world"
        await storage.write("test/file.txt", data)
        assert await storage.read("test/file.txt") == data

    @pytest.mark.asyncio
    async def test_creates_nested_directories(self, storage, base_dir):
        await storage.write("deep/nested/path/file.bin", bytes([1, 2, 3]))
        assert os.path.isfile(os.path.join(base_dir, "deep/nested/path/file.bin"))

    @pytest.mark.asyncio
    async def test_overwrites_existing_values(self, storage):
        await storage.write("key", b"first")
        await storage.write("key", b"second")
        assert await storage.read("key") == b"second"

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_keys(self, storage):
        assert await storage.read("nonexistent/key") is None

    @pytest.mark.asyncio
    async def test_writes_atomically_leaving_no_scratch_file(self, storage, base_dir):
        await storage.write("atomic/test", bytes([1]))
        with open(os.path.join(base_dir, "atomic/test"), "rb") as file:
            assert file.read() == bytes([1])
        scratch = [name for _, _, files in os.walk(base_dir) for name in files if ".__strands_tmp" in name]
        assert scratch == []

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod on a directory does not block rename on Windows")
    @pytest.mark.asyncio
    async def test_cleans_up_scratch_file_on_rename_failure(self, storage, base_dir):
        read_only_dir = os.path.join(base_dir, "readonly")
        os.makedirs(read_only_dir, exist_ok=True)
        with open(os.path.join(read_only_dir, "target"), "w") as file:
            file.write("original")
        os.chmod(read_only_dir, 0o555)
        try:
            with pytest.raises(StorageError):
                await storage.write("readonly/target", bytes([1]))
            os.chmod(read_only_dir, 0o755)
            leftovers = [name for name in os.listdir(read_only_dir) if ".__strands_tmp" in name]
            assert leftovers == []
        finally:
            os.chmod(read_only_dir, 0o755)

    @pytest.mark.asyncio
    async def test_removes_scratch_file_when_rename_fails(self, storage, base_dir):
        # Force the atomic rename to fail *after* the scratch file is written, so the
        # cleanup branch (which removes the leftover scratch file) is exercised.
        with patch("strands.storage.local_file_storage.os.replace", side_effect=OSError("boom")):
            with pytest.raises(StorageError):
                await storage.write("dir/key", bytes([1]))

        scratch = [name for _, _, files in os.walk(base_dir) for name in files if ".__strands_tmp" in name]
        assert scratch == []


class TestDelete:
    @pytest.mark.asyncio
    async def test_removes_an_existing_key(self, storage):
        await storage.write("deleteme", bytes([1]))
        await storage.delete("deleteme")
        assert await storage.read("deleteme") is None

    @pytest.mark.asyncio
    async def test_is_a_no_op_for_missing_keys(self, storage):
        assert await storage.delete("nonexistent") is None


class TestList:
    @pytest.mark.asyncio
    async def test_lists_keys_under_a_prefix(self, storage):
        await storage.write("sessions/a/data.json", bytes([1]))
        await storage.write("sessions/b/data.json", bytes([2]))
        await storage.write("memory/notes.json", bytes([3]))

        tru_keys = await storage.list("sessions/")
        assert tru_keys == ["sessions/a/data.json", "sessions/b/data.json"]

    @pytest.mark.asyncio
    async def test_returns_all_keys_for_empty_prefix(self, storage):
        await storage.write("a", bytes([1]))
        await storage.write("b", bytes([2]))
        assert await storage.list("") == ["a", "b"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_base_directory_does_not_exist(self, tmp_path):
        fresh = LocalFileStorage(str(tmp_path / "does-not-exist"))
        assert await fresh.list("") == []

    @pytest.mark.asyncio
    async def test_excludes_scratch_files(self, storage, base_dir):
        await storage.write("real", bytes([1]))
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "leftover.__strands_tmp"), "w") as file:
            file.write("garbage")

        assert "leftover.__strands_tmp" not in await storage.list("")

    @pytest.mark.asyncio
    async def test_does_not_exclude_user_dot_tmp_files(self, storage):
        await storage.write("notes.tmp", bytes([1]))
        assert "notes.tmp" in await storage.list("")

    @pytest.mark.asyncio
    async def test_returns_keys_sorted_lexicographically(self, storage):
        await storage.write("c", bytes([3]))
        await storage.write("a", bytes([1]))
        await storage.write("b", bytes([2]))
        assert await storage.list("") == ["a", "b", "c"]
