"""Tests for the LocalSandbox implementation."""

import asyncio
import os
from pathlib import Path

import pytest

from strands.sandbox.base import ExecutionResult, StreamChunk, StreamType, FileInfo, Sandbox
from strands.sandbox.local import LocalSandbox


class TestLocalSandboxInit:
    def test_default_working_dir(self) -> None:
        sandbox = LocalSandbox()
        assert sandbox.working_dir == os.getcwd()

    def test_custom_working_dir(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        assert sandbox.working_dir == str(tmp_path)

    def test_extends_sandbox_directly(self) -> None:
        """LocalSandbox extends Sandbox directly, not ShellBasedSandbox."""
        sandbox = LocalSandbox()
        assert isinstance(sandbox, Sandbox)

    def test_does_not_extend_shell_based_sandbox(self) -> None:
        """LocalSandbox must NOT inherit from ShellBasedSandbox."""
        from strands.sandbox.shell_based import ShellBasedSandbox

        sandbox = LocalSandbox()
        assert not isinstance(sandbox, ShellBasedSandbox)


class TestLocalSandboxResolvePath:
    def test_resolve_relative_path(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        resolved = sandbox._resolve_path("subdir/file.txt")
        expected = Path(str(tmp_path)) / "subdir" / "file.txt"
        assert resolved == expected

    def test_resolve_absolute_path(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        abs_path = str(Path(str(tmp_path)) / "absolute" / "path.txt")
        resolved = sandbox._resolve_path(abs_path)
        assert str(resolved) == abs_path


class TestLocalSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_echo(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_streams_chunks(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks: list[StreamChunk | ExecutionResult] = []
        async for chunk in sandbox.execute_streaming("echo line1 && echo line2"):
            chunks.append(chunk)
        str_chunks = [c.data for c in chunks if isinstance(c, StreamChunk)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert len(str_chunks) >= 1
        combined = "".join(str_chunks)
        assert "line1" in combined
        assert "line2" in combined

    @pytest.mark.asyncio
    async def test_execute_failure(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute("exit 42")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_execute_stderr(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute("echo error >&2")
        assert result.exit_code == 0
        assert result.stderr.strip() == "error"

    @pytest.mark.asyncio
    async def test_execute_uses_working_dir(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # Use python to print cwd instead of 'pwd' for cross-platform compatibility.
        # On Windows, 'pwd' (via Git Bash) returns MSYS-style paths (/c/Users/...)
        # which don't match the native Windows path (C:\Users\...).
        result = await sandbox.execute('python -c "import os; print(os.getcwd())"')
        assert result.exit_code == 0
        assert os.path.normpath(result.stdout.strip()) == os.path.normpath(str(tmp_path))

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(asyncio.TimeoutError):
            await sandbox.execute("sleep 10", timeout=1)

    @pytest.mark.asyncio
    async def test_execute_no_timeout(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute("echo fast", timeout=None)
        assert result.exit_code == 0
        assert result.stdout.strip() == "fast"
    @pytest.mark.asyncio
    async def test_execute_long_output_without_newlines(self, tmp_path: object) -> None:
        """Bug 2 fix: long output lines (>64KB) no longer crash."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute("python3 -c \"import sys; sys.stdout.write('A' * 131072)\"")
        assert result.exit_code == 0
        assert len(result.stdout) == 131072


class TestLocalSandboxExecuteCode:
    @pytest.mark.asyncio
    async def test_execute_python_code(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute_code("print('hello from python')", language="python")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello from python"

    @pytest.mark.asyncio
    async def test_execute_code_streams(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks: list[StreamChunk | ExecutionResult] = []
        async for chunk in sandbox.execute_code_streaming("print('line1')\nprint('line2')", language="python"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        combined = "".join(c.data for c in chunks if isinstance(c, StreamChunk))
        assert "line1" in combined
        assert "line2" in combined

    @pytest.mark.asyncio
    async def test_execute_python_code_error(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute_code("raise ValueError('test error')", language="python")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_python_multiline(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = "x = 42\nprint(f'x = {x}')"
        result = await sandbox.execute_code(code, language="python")
        assert result.exit_code == 0
        assert "x = 42" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_rejects_unsafe_language(self, tmp_path: object) -> None:
        """Language parameter with shell metacharacters raises ValueError."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unsafe characters"):
            await sandbox.execute_code("print(1)", language="python; rm -rf /")

    @pytest.mark.asyncio
    async def test_execute_code_accepts_valid_languages(self, tmp_path: object) -> None:
        """Valid language names with dots, hyphens, underscores pass validation."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # python3.12-like names should be accepted
        # (may fail to execute if interpreter not found, but validation passes)
        with pytest.raises(ValueError, match="unsafe characters"):
            await sandbox.execute_code("1", language="python; echo pwned")
        # These should NOT raise ValueError (may raise FileNotFoundError at exec time)
        try:
            await sandbox.execute_code("1", language="python3.12")
        except Exception as e:
            # FileNotFoundError or non-zero exit code is expected, NOT ValueError
            assert not isinstance(e, ValueError)

    @pytest.mark.asyncio
    async def test_execute_code_uses_subprocess_exec(self, tmp_path: object) -> None:
        """Verify code is passed directly to interpreter, not through shell."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # Code with shell metacharacters should be passed literally to python
        code = "import sys; print(sys.argv)"
        result = await sandbox.execute_code(code, language="python")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_code_uses_working_dir(self, tmp_path: object) -> None:
        """Code execution should use the sandbox working directory."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute_code("import os; print(os.getcwd())", language="python")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_execute_code_with_quotes(self, tmp_path: object) -> None:
        """Code with all types of quotes should work (no shell quoting needed)."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = """
x = "hello 'world'"
y = 'hello "world"'
print(x)
print(y)
"""
        result = await sandbox.execute_code(code, language="python")
        assert result.exit_code == 0
        assert "hello 'world'" in result.stdout
        assert 'hello "world"' in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_with_backslashes(self, tmp_path: object) -> None:
        """Code with backslashes should work (no shell escaping needed)."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = 'print("path\\\\to\\\\file")'
        result = await sandbox.execute_code(code, language="python")
        assert result.exit_code == 0
        assert "path\\to\\file" in result.stdout


class TestLocalSandboxFileOps:
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("test.txt", b"hello world")
        content = await sandbox.read_file("test.txt")
        assert content == b"hello world"

    @pytest.mark.asyncio
    async def test_read_file_absolute_path(self, tmp_path: object) -> None:
        test_file = tmp_path / "abs_test.txt"  # type: ignore[operator]
        test_file.write_bytes(b"absolute content")
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = await sandbox.read_file(str(test_file))
        assert content == b"absolute content"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("subdir/nested/test.txt", b"nested content")
        content = await sandbox.read_file("subdir/nested/test.txt")
        assert content == b"nested content"

    @pytest.mark.asyncio
    async def test_write_file_absolute_path(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        abs_path = str(tmp_path / "abs_write.txt")  # type: ignore[operator]
        await sandbox.write_file(abs_path, b"absolute write")
        content = await sandbox.read_file(abs_path)
        assert content == b"absolute write"

    @pytest.mark.asyncio
    async def test_write_file_unicode(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_text("unicode.txt", "héllo wörld 🌍")
        content = await sandbox.read_text("unicode.txt")
        assert content == "héllo wörld 🌍"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path: object) -> None:
        (tmp_path / "file1.txt").write_text("a")  # type: ignore[operator]
        (tmp_path / "file2.txt").write_text("b")  # type: ignore[operator]
        (tmp_path / "file3.py").write_text("c")  # type: ignore[operator]

        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files(".")
        names = sorted([f.name for f in files])
        assert names == ["file1.txt", "file2.txt", "file3.py"]
        # All should be FileInfo instances
        for f in files:
            assert isinstance(f, FileInfo)
            assert f.is_dir is False

    @pytest.mark.asyncio
    async def test_list_files_includes_hidden_files(self, tmp_path: object) -> None:
        """list_files uses os.listdir which includes hidden files (unlike ls -1)."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("visible.txt", b"visible")
        await sandbox.write_file(".hidden", b"hidden")

        files = await sandbox.list_files(".")
        names = [f.name for f in files]
        assert "visible.txt" in names
        assert ".hidden" in names  # Native Python includes dotfiles!

    @pytest.mark.asyncio
    async def test_list_files_sorted(self, tmp_path: object) -> None:
        """list_files returns sorted results for deterministic ordering."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("zebra.txt", b"z")
        await sandbox.write_file("apple.txt", b"a")
        await sandbox.write_file("mango.txt", b"m")

        files = await sandbox.list_files(".")
        names = [f.name for f in files]
        assert names == ["apple.txt", "mango.txt", "zebra.txt"]

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, tmp_path: object) -> None:
        empty_dir = tmp_path / "empty"  # type: ignore[operator]
        empty_dir.mkdir()
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        files = await sandbox.list_files("empty")
        assert files == []  # Empty list of FileInfo

    @pytest.mark.asyncio
    async def test_list_files_not_found(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.list_files("nonexistent")
class TestLocalSandboxExecuteCodeErrorHandling:
    """Tests for execute_code error handling (FileNotFoundError fix)."""

    @pytest.mark.asyncio
    async def test_execute_code_nonexistent_language_returns_exit_127(self, tmp_path: object) -> None:
        """execute_code with non-existent language returns exit 127 instead of crashing."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox.execute_code("1", language="nonexistent-lang-12345")
        assert result.exit_code == 127
        assert "not found" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_code_nonexistent_language_streams_result(self, tmp_path: object) -> None:
        """execute_code yields an ExecutionResult even for non-existent languages."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        chunks: list = []
        async for chunk in sandbox.execute_code_streaming("1", language="nonexistent-lang-xyz"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ExecutionResult)
        assert chunks[0].exit_code == 127


class TestLocalSandboxBinaryIO:
    """Tests for binary file I/O (bytes-native read/write)."""

    @pytest.mark.asyncio
    async def test_write_and_read_binary(self, tmp_path: object) -> None:
        """Binary content (e.g., PNG header) round-trips correctly."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        await sandbox.write_file("image.png", binary_content)
        read_back = await sandbox.read_file("image.png")
        assert read_back == binary_content

    @pytest.mark.asyncio
    async def test_read_text_convenience(self, tmp_path: object) -> None:
        """read_text decodes bytes to string."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("test.txt", b"hello world")
        text = await sandbox.read_text("test.txt")
        assert text == "hello world"
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_write_text_convenience(self, tmp_path: object) -> None:
        """write_text encodes string to bytes."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_text("test.txt", "hello world")
        read_back = await sandbox.read_file("test.txt")
        assert read_back == b"hello world"

    @pytest.mark.asyncio
    async def test_read_text_unicode_decode_error(self, tmp_path: object) -> None:
        """read_text raises UnicodeDecodeError for binary content."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("binary.bin", b"\x89PNG\xff\xfe")
        with pytest.raises(UnicodeDecodeError):
            await sandbox.read_text("binary.bin")


class TestLocalSandboxFileInfoMetadata:
    """Tests for structured FileInfo returns from list_files."""

    @pytest.mark.asyncio
    async def test_list_files_directories_have_is_dir_true(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        (tmp_path / "subdir").mkdir()  # type: ignore[operator]
        (tmp_path / "file.txt").write_bytes(b"content")  # type: ignore[operator]
        files = await sandbox.list_files(".")
        dir_entry = next(f for f in files if f.name == "subdir")
        file_entry = next(f for f in files if f.name == "file.txt")
        assert dir_entry.is_dir is True
        assert file_entry.is_dir is False

    @pytest.mark.asyncio
    async def test_list_files_reports_file_size(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        (tmp_path / "sized.txt").write_bytes(b"x" * 42)  # type: ignore[operator]
        files = await sandbox.list_files(".")
        entry = files[0]
        assert entry.name == "sized.txt"
        assert entry.size == 42
