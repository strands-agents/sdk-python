"""Tests for the LocalSandbox implementation."""

import asyncio
import os

import pytest

from strands.sandbox.base import ExecutionResult
from strands.sandbox.local import LocalSandbox


class TestLocalSandboxInit:
    def test_default_working_dir(self):
        sandbox = LocalSandbox()
        assert sandbox.working_dir == os.getcwd()

    def test_custom_working_dir(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        assert sandbox.working_dir == str(tmp_path)


class TestLocalSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_echo(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo hello")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_streams_lines(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks = []
        async for chunk in sandbox.execute("echo line1 && echo line2"):
            chunks.append(chunk)
        # Should have string lines and a final ExecutionResult
        str_chunks = [c for c in chunks if isinstance(c, str)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert len(str_chunks) >= 2
        assert "line1\n" in str_chunks
        assert "line2\n" in str_chunks

    @pytest.mark.asyncio
    async def test_execute_failure(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("exit 42")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_execute_stderr(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo error >&2")
        assert result.exit_code == 0
        assert result.stderr.strip() == "error"

    @pytest.mark.asyncio
    async def test_execute_uses_working_dir(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("pwd")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(asyncio.TimeoutError):
            await sandbox._execute_to_result("sleep 10", timeout=1)

    @pytest.mark.asyncio
    async def test_execute_no_timeout(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo fast", timeout=None)
        assert result.exit_code == 0
        assert result.stdout.strip() == "fast"

    @pytest.mark.asyncio
    async def test_auto_start(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        assert not sandbox._started
        result = await sandbox._execute_to_result("echo hello")
        assert sandbox._started
        assert result.exit_code == 0


class TestLocalSandboxExecuteCode:
    @pytest.mark.asyncio
    async def test_execute_python_code(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_code_to_result("print('hello from python')")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello from python"

    @pytest.mark.asyncio
    async def test_execute_code_streams(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks = []
        async for chunk in sandbox.execute_code("print('line1')\\nprint('line2')"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_python_code_error(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_code_to_result("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_python_multiline(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = "x = 42\nprint(f'x = {x}')"
        result = await sandbox._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "x = 42" in result.stdout


class TestLocalSandboxFileOps:
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("test.txt", "hello world")
        content = await sandbox.read_file("test.txt")
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_absolute_path(self, tmp_path):
        test_file = tmp_path / "abs_test.txt"
        test_file.write_text("absolute content")
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = await sandbox.read_file(str(test_file))
        assert content == "absolute content"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("subdir/nested/test.txt", "nested content")
        content = await sandbox.read_file("subdir/nested/test.txt")
        assert content == "nested content"

    @pytest.mark.asyncio
    async def test_write_file_absolute_path(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        abs_path = str(tmp_path / "abs_write.txt")
        await sandbox.write_file(abs_path, "absolute write")
        content = await sandbox.read_file(abs_path)
        assert content == "absolute write"

    @pytest.mark.asyncio
    async def test_write_file_unicode(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("unicode.txt", "héllo wörld 🌍")
        content = await sandbox.read_file("unicode.txt")
        assert content == "héllo wörld 🌍"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")
        (tmp_path / "file3.py").write_text("c")

        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files(".")
        assert sorted(files) == ["file1.txt", "file2.txt", "file3.py"]

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files("empty")
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_not_found(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.list_files("nonexistent")


class TestLocalSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        assert sandbox._started
        await sandbox.stop()
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        async with LocalSandbox(working_dir=str(tmp_path)) as sandbox:
            result = await sandbox._execute_to_result("echo context")
            assert result.stdout.strip() == "context"
