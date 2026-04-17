"""Tests for the LocalWorkspace implementation."""

import asyncio
import os

import pytest

from strands.workspace.base import ExecutionResult, FileInfo, Workspace
from strands.workspace.local import LocalWorkspace


class TestLocalWorkspaceInit:
    def test_default_working_dir(self) -> None:
        workspace = LocalWorkspace()
        assert workspace.working_dir == os.getcwd()

    def test_custom_working_dir(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        assert workspace.working_dir == str(tmp_path)

    def test_extends_workspace_directly(self) -> None:
        """LocalWorkspace extends Workspace directly, not ShellBasedWorkspace."""
        workspace = LocalWorkspace()
        assert isinstance(workspace, Workspace)

    def test_does_not_extend_shell_based_workspace(self) -> None:
        """LocalWorkspace must NOT inherit from ShellBasedWorkspace."""
        from strands.workspace.shell_based import ShellBasedWorkspace

        workspace = LocalWorkspace()
        assert not isinstance(workspace, ShellBasedWorkspace)


class TestLocalWorkspaceResolvePath:
    def test_resolve_relative_path(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        resolved = workspace._resolve_path("subdir/file.txt")
        assert str(resolved) == os.path.join(str(tmp_path), "subdir/file.txt")

    def test_resolve_absolute_path(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        resolved = workspace._resolve_path("/absolute/path.txt")
        assert str(resolved) == "/absolute/path.txt"


class TestLocalWorkspaceExecute:
    @pytest.mark.asyncio
    async def test_execute_echo(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("echo hello")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_streams_chunks(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        chunks: list[str | ExecutionResult] = []
        async for chunk in workspace.execute("echo line1 && echo line2"):
            chunks.append(chunk)
        str_chunks = [c for c in chunks if isinstance(c, str)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert len(str_chunks) >= 1
        combined = "".join(str_chunks)
        assert "line1" in combined
        assert "line2" in combined

    @pytest.mark.asyncio
    async def test_execute_failure(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("exit 42")
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_execute_stderr(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("echo error >&2")
        assert result.exit_code == 0
        assert result.stderr.strip() == "error"

    @pytest.mark.asyncio
    async def test_execute_uses_working_dir(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("pwd")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(asyncio.TimeoutError):
            await workspace._execute_to_result("sleep 10", timeout=1)

    @pytest.mark.asyncio
    async def test_execute_no_timeout(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("echo fast", timeout=None)
        assert result.exit_code == 0
        assert result.stdout.strip() == "fast"

    @pytest.mark.asyncio
    async def test_auto_start(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        assert not workspace._started
        result = await workspace._execute_to_result("echo hello")
        assert workspace._started
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_long_output_without_newlines(self, tmp_path: object) -> None:
        """Bug 2 fix: long output lines (>64KB) no longer crash."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("python3 -c \"import sys; sys.stdout.write('A' * 131072)\"")
        assert result.exit_code == 0
        assert len(result.stdout) == 131072


class TestLocalWorkspaceExecuteCode:
    @pytest.mark.asyncio
    async def test_execute_python_code(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_code_to_result("print('hello from python')")
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello from python"

    @pytest.mark.asyncio
    async def test_execute_code_streams(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        chunks: list[str | ExecutionResult] = []
        async for chunk in workspace.execute_code("print('line1')\nprint('line2')"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        combined = "".join(c for c in chunks if isinstance(c, str))
        assert "line1" in combined
        assert "line2" in combined

    @pytest.mark.asyncio
    async def test_execute_python_code_error(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_code_to_result("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_python_multiline(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        code = "x = 42\nprint(f'x = {x}')"
        result = await workspace._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "x = 42" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_rejects_unsafe_language(self, tmp_path: object) -> None:
        """Language parameter with shell metacharacters raises ValueError."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unsafe characters"):
            await workspace._execute_code_to_result("print(1)", language="python; rm -rf /")

    @pytest.mark.asyncio
    async def test_execute_code_accepts_valid_languages(self, tmp_path: object) -> None:
        """Valid language names with dots, hyphens, underscores pass validation."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        # python3.12-like names should be accepted
        # (may fail to execute if interpreter not found, but validation passes)
        with pytest.raises(ValueError, match="unsafe characters"):
            await workspace._execute_code_to_result("1", language="python; echo pwned")
        # These should NOT raise ValueError (may raise FileNotFoundError at exec time)
        try:
            await workspace._execute_code_to_result("1", language="python3.12")
        except Exception as e:
            # FileNotFoundError or non-zero exit code is expected, NOT ValueError
            assert not isinstance(e, ValueError)

    @pytest.mark.asyncio
    async def test_execute_code_uses_subprocess_exec(self, tmp_path: object) -> None:
        """Verify code is passed directly to interpreter, not through shell."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        # Code with shell metacharacters should be passed literally to python
        code = "import sys; print(sys.argv)"
        result = await workspace._execute_code_to_result(code)
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_code_uses_working_dir(self, tmp_path: object) -> None:
        """Code execution should use the workspace working directory."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_code_to_result("import os; print(os.getcwd())")
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path)

    @pytest.mark.asyncio
    async def test_execute_code_with_quotes(self, tmp_path: object) -> None:
        """Code with all types of quotes should work (no shell quoting needed)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        code = """
x = "hello 'world'"
y = 'hello "world"'
print(x)
print(y)
"""
        result = await workspace._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "hello 'world'" in result.stdout
        assert 'hello "world"' in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_with_backslashes(self, tmp_path: object) -> None:
        """Code with backslashes should work (no shell escaping needed)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        code = 'print("path\\\\to\\\\file")'
        result = await workspace._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "path\\to\\file" in result.stdout


class TestLocalWorkspaceFileOps:
    @pytest.mark.asyncio
    async def test_write_and_read_file(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("test.txt", b"hello world")
        content = await workspace.read_file("test.txt")
        assert content == b"hello world"

    @pytest.mark.asyncio
    async def test_read_file_absolute_path(self, tmp_path: object) -> None:
        test_file = tmp_path / "abs_test.txt"  # type: ignore[operator]
        test_file.write_bytes(b"absolute content")
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        content = await workspace.read_file(str(test_file))
        assert content == b"absolute content"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await workspace.read_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_write_file_creates_directories(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("subdir/nested/test.txt", b"nested content")
        content = await workspace.read_file("subdir/nested/test.txt")
        assert content == b"nested content"

    @pytest.mark.asyncio
    async def test_write_file_absolute_path(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        abs_path = str(tmp_path / "abs_write.txt")  # type: ignore[operator]
        await workspace.write_file(abs_path, b"absolute write")
        content = await workspace.read_file(abs_path)
        assert content == b"absolute write"

    @pytest.mark.asyncio
    async def test_write_file_unicode(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_text("unicode.txt", "héllo wörld 🌍")
        content = await workspace.read_text("unicode.txt")
        assert content == "héllo wörld 🌍"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path: object) -> None:
        (tmp_path / "file1.txt").write_text("a")  # type: ignore[operator]
        (tmp_path / "file2.txt").write_text("b")  # type: ignore[operator]
        (tmp_path / "file3.py").write_text("c")  # type: ignore[operator]

        workspace = LocalWorkspace(working_dir=str(tmp_path))
        files = await workspace.list_files(".")
        names = sorted([f.name for f in files])
        assert names == ["file1.txt", "file2.txt", "file3.py"]
        # All should be FileInfo instances
        for f in files:
            assert isinstance(f, FileInfo)
            assert f.is_dir is False

    @pytest.mark.asyncio
    async def test_list_files_includes_hidden_files(self, tmp_path: object) -> None:
        """list_files uses os.listdir which includes hidden files (unlike ls -1)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("visible.txt", b"visible")
        await workspace.write_file(".hidden", b"hidden")

        files = await workspace.list_files(".")
        names = [f.name for f in files]
        assert "visible.txt" in names
        assert ".hidden" in names  # Native Python includes dotfiles!

    @pytest.mark.asyncio
    async def test_list_files_sorted(self, tmp_path: object) -> None:
        """list_files returns sorted results for deterministic ordering."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("zebra.txt", b"z")
        await workspace.write_file("apple.txt", b"a")
        await workspace.write_file("mango.txt", b"m")

        files = await workspace.list_files(".")
        names = [f.name for f in files]
        assert names == ["apple.txt", "mango.txt", "zebra.txt"]

    @pytest.mark.asyncio
    async def test_list_files_empty_dir(self, tmp_path: object) -> None:
        empty_dir = tmp_path / "empty"  # type: ignore[operator]
        empty_dir.mkdir()
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        files = await workspace.list_files("empty")
        assert files == []  # Empty list of FileInfo

    @pytest.mark.asyncio
    async def test_list_files_not_found(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await workspace.list_files("nonexistent")


class TestLocalWorkspaceLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        assert workspace._started
        await workspace.stop()
        assert not workspace._started

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: object) -> None:
        async with LocalWorkspace(working_dir=str(tmp_path)) as workspace:
            result = await workspace._execute_to_result("echo context")
            assert result.stdout.strip() == "context"


class TestLocalWorkspaceWorkingDirValidation:
    """Tests for Bug 3 fix: non-existent working directory handling."""

    @pytest.mark.asyncio
    async def test_start_creates_nonexistent_working_dir(self, tmp_path: object) -> None:
        new_dir = str(tmp_path / "new" / "deep" / "dir")  # type: ignore[operator]
        workspace = LocalWorkspace(working_dir=new_dir)
        await workspace.start()
        assert os.path.isdir(new_dir)
        assert workspace._started

    @pytest.mark.asyncio
    async def test_auto_start_creates_working_dir(self, tmp_path: object) -> None:
        new_dir = str(tmp_path / "auto_created")  # type: ignore[operator]
        workspace = LocalWorkspace(working_dir=new_dir)
        result = await workspace._execute_to_result("echo hello")
        assert result.exit_code == 0
        assert os.path.isdir(new_dir)

    @pytest.mark.asyncio
    async def test_start_raises_if_path_is_a_file(self, tmp_path: object) -> None:
        file_path = tmp_path / "not_a_dir"  # type: ignore[operator]
        file_path.write_text("i am a file")
        workspace = LocalWorkspace(working_dir=str(file_path))
        with pytest.raises(NotADirectoryError, match="not a directory"):
            await workspace.start()

    @pytest.mark.asyncio
    async def test_existing_working_dir_is_fine(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        assert workspace._started


class TestLocalWorkspaceExecuteCodeErrorHandling:
    """Tests for execute_code error handling (FileNotFoundError fix)."""

    @pytest.mark.asyncio
    async def test_execute_code_nonexistent_language_returns_exit_127(self, tmp_path: object) -> None:
        """execute_code with non-existent language returns exit 127 instead of crashing."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_code_to_result("1", language="nonexistent-lang-12345")
        assert result.exit_code == 127
        assert "not found" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_code_nonexistent_language_streams_result(self, tmp_path: object) -> None:
        """execute_code yields an ExecutionResult even for non-existent languages."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        chunks: list = []
        async for chunk in workspace.execute_code("1", language="nonexistent-lang-xyz"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ExecutionResult)
        assert chunks[0].exit_code == 127


class TestLocalWorkspaceBinaryIO:
    """Tests for binary file I/O (bytes-native read/write)."""

    @pytest.mark.asyncio
    async def test_write_and_read_binary(self, tmp_path: object) -> None:
        """Binary content (e.g., PNG header) round-trips correctly."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        await workspace.write_file("image.png", binary_content)
        read_back = await workspace.read_file("image.png")
        assert read_back == binary_content

    @pytest.mark.asyncio
    async def test_read_text_convenience(self, tmp_path: object) -> None:
        """read_text decodes bytes to string."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("test.txt", b"hello world")
        text = await workspace.read_text("test.txt")
        assert text == "hello world"
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_write_text_convenience(self, tmp_path: object) -> None:
        """write_text encodes string to bytes."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_text("test.txt", "hello world")
        read_back = await workspace.read_file("test.txt")
        assert read_back == b"hello world"

    @pytest.mark.asyncio
    async def test_read_text_unicode_decode_error(self, tmp_path: object) -> None:
        """read_text raises UnicodeDecodeError for binary content."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("binary.bin", b"\x89PNG\xff\xfe")
        with pytest.raises(UnicodeDecodeError):
            await workspace.read_text("binary.bin")


class TestLocalWorkspaceFileInfoMetadata:
    """Tests for structured FileInfo returns from list_files."""

    @pytest.mark.asyncio
    async def test_list_files_directories_have_is_dir_true(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        (tmp_path / "subdir").mkdir()  # type: ignore[operator]
        (tmp_path / "file.txt").write_bytes(b"content")  # type: ignore[operator]
        files = await workspace.list_files(".")
        dir_entry = next(f for f in files if f.name == "subdir")
        file_entry = next(f for f in files if f.name == "file.txt")
        assert dir_entry.is_dir is True
        assert file_entry.is_dir is False

    @pytest.mark.asyncio
    async def test_list_files_reports_file_size(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        (tmp_path / "sized.txt").write_bytes(b"x" * 42)  # type: ignore[operator]
        files = await workspace.list_files(".")
        entry = files[0]
        assert entry.name == "sized.txt"
        assert entry.size == 42
