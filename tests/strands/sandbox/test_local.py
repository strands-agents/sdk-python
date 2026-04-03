"""Tests for the LocalSandbox implementation."""

import asyncio
import os

import pytest

from strands.sandbox.base import ExecutionResult, ShellBasedSandbox
from strands.sandbox.local import LocalSandbox


class TestLocalSandboxInit:
    def test_default_working_dir(self) -> None:
        sandbox = LocalSandbox()
        assert sandbox.working_dir == os.getcwd()

    def test_custom_working_dir(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        assert sandbox.working_dir == str(tmp_path)

    def test_inherits_shell_based_sandbox(self) -> None:
        sandbox = LocalSandbox()
        assert isinstance(sandbox, ShellBasedSandbox)


class TestLocalSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_echo(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo hello")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_streams_chunks(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks: list[str | ExecutionResult] = []
        async for chunk in sandbox.execute("echo line1 && echo line2"):
            chunks.append(chunk)
        str_chunks = [c for c in chunks if isinstance(c, str)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert len(str_chunks) >= 1
        # All output should be present in the concatenated string chunks
        combined = "".join(str_chunks)
        assert "line1" in combined
        assert "line2" in combined

    @pytest.mark.asyncio
    async def test_execute_failure(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("exit 42")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_execute_stderr(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo error >&2")
        assert result.exit_code == 0
        assert result.stderr.strip() == "error"

    @pytest.mark.asyncio
    async def test_execute_uses_working_dir(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("pwd")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(asyncio.TimeoutError):
            await sandbox._execute_to_result("sleep 10", timeout=1)

    @pytest.mark.asyncio
    async def test_execute_no_timeout(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("echo fast", timeout=None)
        assert result.exit_code == 0
        assert result.stdout.strip() == "fast"

    @pytest.mark.asyncio
    async def test_auto_start(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        assert not sandbox._started
        result = await sandbox._execute_to_result("echo hello")
        assert sandbox._started
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_long_output_without_newlines(self, tmp_path: object) -> None:
        """Bug 2 fix: long output lines (>64KB) no longer crash."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # Generate 128KB of output without any newline
        result = await sandbox._execute_to_result(
            "python3 -c \"import sys; sys.stdout.write('A' * 131072)\""
        )
        assert result.exit_code == 0
        assert len(result.stdout) == 131072


class TestLocalSandboxExecuteCode:
    @pytest.mark.asyncio
    async def test_execute_python_code(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_code_to_result("print('hello from python')")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello from python"

    @pytest.mark.asyncio
    async def test_execute_code_streams(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks: list[str | ExecutionResult] = []
        async for chunk in sandbox.execute_code("print('line1')\\nprint('line2')"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_python_code_error(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_code_to_result("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_python_multiline(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = "x = 42\nprint(f'x = {x}')"
        result = await sandbox._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "x = 42" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_rejects_language_injection(self, tmp_path: object) -> None:
        """Bug 1 fix: language parameter injection is blocked."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(ValueError):
            await sandbox._execute_code_to_result("print(1)", language="python; rm -rf /")


class TestLocalSandboxFileOps:
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("test.txt", "hello world")
        content = await sandbox.read_file("test.txt")
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_read_file_absolute_path(self, tmp_path: object) -> None:
        test_file = tmp_path / "abs_test.txt"  # type: ignore[operator]
        test_file.write_text("absolute content")
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = await sandbox.read_file(str(test_file))
        assert content == "absolute content"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("subdir/nested/test.txt", "nested content")
        content = await sandbox.read_file("subdir/nested/test.txt")
        assert content == "nested content"

    @pytest.mark.asyncio
    async def test_write_file_absolute_path(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        abs_path = str(tmp_path / "abs_write.txt")  # type: ignore[operator]
        await sandbox.write_file(abs_path, "absolute write")
        content = await sandbox.read_file(abs_path)
        assert content == "absolute write"

    @pytest.mark.asyncio
    async def test_write_file_unicode(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("unicode.txt", "héllo wörld 🌍")
        content = await sandbox.read_file("unicode.txt")
        assert content == "héllo wörld 🌍"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path: object) -> None:
        (tmp_path / "file1.txt").write_text("a")  # type: ignore[operator]
        (tmp_path / "file2.txt").write_text("b")  # type: ignore[operator]
        (tmp_path / "file3.py").write_text("c")  # type: ignore[operator]

        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files(".")
        assert sorted(files) == ["file1.txt", "file2.txt", "file3.py"]

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, tmp_path: object) -> None:
        empty_dir = tmp_path / "empty"  # type: ignore[operator]
        empty_dir.mkdir()
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files("empty")
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_not_found(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.list_files("nonexistent")


class TestLocalSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        assert sandbox._started
        await sandbox.stop()
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: object) -> None:
        async with LocalSandbox(working_dir=str(tmp_path)) as sandbox:
            result = await sandbox._execute_to_result("echo context")
            assert result.stdout.strip() == "context"


class TestLocalSandboxWorkingDirValidation:
    """Tests for Bug 3 fix: non-existent working directory handling."""

    @pytest.mark.asyncio
    async def test_start_creates_nonexistent_working_dir(self, tmp_path: object) -> None:
        """start() creates a non-existent working directory."""
        new_dir = str(tmp_path / "new" / "deep" / "dir")  # type: ignore[operator]
        sandbox = LocalSandbox(working_dir=new_dir)
        await sandbox.start()
        assert os.path.isdir(new_dir)
        assert sandbox._started

    @pytest.mark.asyncio
    async def test_auto_start_creates_working_dir(self, tmp_path: object) -> None:
        """Auto-start on first execute also creates the directory."""
        new_dir = str(tmp_path / "auto_created")  # type: ignore[operator]
        sandbox = LocalSandbox(working_dir=new_dir)
        result = await sandbox._execute_to_result("echo hello")
        assert result.exit_code == 0
        assert os.path.isdir(new_dir)

    @pytest.mark.asyncio
    async def test_start_raises_if_path_is_a_file(self, tmp_path: object) -> None:
        """start() raises NotADirectoryError if working_dir is a file."""
        file_path = tmp_path / "not_a_dir"  # type: ignore[operator]
        file_path.write_text("i am a file")
        sandbox = LocalSandbox(working_dir=str(file_path))
        with pytest.raises(NotADirectoryError, match="not a directory"):
            await sandbox.start()

    @pytest.mark.asyncio
    async def test_existing_working_dir_is_fine(self, tmp_path: object) -> None:
        """start() succeeds with an existing directory."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        assert sandbox._started
