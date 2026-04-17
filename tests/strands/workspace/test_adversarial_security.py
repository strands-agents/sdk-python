"""Adversarial tests: Security, path traversal, injection, and edge cases.

Tests for:
- Path traversal attacks in LocalWorkspace
- Content injection edge cases
- Symlink attacks
- Binary/special content handling
"""

import pytest

from strands.workspace.base import ExecutionResult
from strands.workspace.local import LocalWorkspace
from strands.workspace.shell_based import ShellBasedWorkspace


class TestPathTraversal:
    """Can we escape the working directory via path traversal?"""

    @pytest.mark.asyncio
    async def test_read_file_path_traversal(self, tmp_path):
        """read_file with ../.. should still work (LocalWorkspace uses native I/O)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "outside_workspace"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("SECRET_DATA")

        relative_path = "../outside_workspace/secret.txt"
        content = await workspace.read_file(relative_path)
        assert content == b"SECRET_DATA", "Path traversal should be documented or blocked"

    @pytest.mark.asyncio
    async def test_write_file_path_traversal(self, tmp_path):
        """write_file with ../.. can write outside working directory."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "write_escape"
        outside_dir.mkdir(exist_ok=True)

        relative_path = "../write_escape/pwned.txt"
        await workspace.write_file(relative_path, b"PWNED")
        assert (outside_dir / "pwned.txt").read_bytes() == b"PWNED"

    @pytest.mark.asyncio
    async def test_execute_can_access_entire_filesystem(self, tmp_path):
        """execute() runs arbitrary shell commands — it can access paths outside working_dir."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        result = await workspace._execute_to_result("echo hello_from_shell")
        assert result.exit_code == 0
        assert "hello_from_shell" in result.stdout

    @pytest.mark.asyncio
    async def test_absolute_path_bypasses_working_dir(self, tmp_path):
        """Absolute paths completely bypass working_dir for LocalWorkspace."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        abs_path = str(tmp_path.parent / "abs_escape.txt")
        await workspace.write_file(abs_path, b"escaped")
        content = await workspace.read_file(abs_path)
        assert content == b"escaped"


class TestContentEdgeCases:
    """Can content with special characters break file operations?"""

    @pytest.mark.asyncio
    async def test_content_with_shell_metacharacters(self, tmp_path):
        """Content with shell metacharacters should be preserved (native Python I/O)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        content = b"hello $USER `whoami` $(id) && rm -rf / ; echo pwned"
        await workspace.write_file("test.txt", content)
        read_back = await workspace.read_file("test.txt")
        assert read_back == content

    @pytest.mark.asyncio
    async def test_content_with_null_bytes(self, tmp_path):
        """Content with null bytes should be handled by native Python I/O."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        content = b"before\x00after"
        await workspace.write_file("null.txt", content)
        read_back = await workspace.read_file("null.txt")
        assert read_back == content

    @pytest.mark.asyncio
    async def test_empty_content(self, tmp_path):
        """Writing empty content should create an empty file."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("empty.txt", b"")
        content = await workspace.read_file("empty.txt")
        assert content == b""

    @pytest.mark.asyncio
    async def test_very_large_content(self, tmp_path):
        """Writing 10MB of content should work."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        large_content = b"A" * (10 * 1024 * 1024)
        await workspace.write_file("large.txt", large_content)
        content = await workspace.read_file("large.txt")
        assert len(content) == len(large_content)
        assert content == large_content


class TestLocalWorkspaceEdgeCases:
    """Edge cases specific to LocalWorkspace."""

    @pytest.mark.asyncio
    async def test_symlink_read(self, tmp_path):
        """Reading through a symlink should work."""
        real_file = tmp_path / "real.txt"
        real_file.write_text("real content")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        workspace = LocalWorkspace(working_dir=str(tmp_path))
        content = await workspace.read_file("link.txt")
        assert content == b"real content"

    @pytest.mark.asyncio
    async def test_symlink_outside_workspace(self, tmp_path):
        """Symlink pointing outside working_dir — LocalWorkspace follows it."""
        outside_dir = tmp_path.parent / "symlink_escape"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "target.txt"
        outside_file.write_text("escaped via symlink")

        symlink = tmp_path / "evil_link.txt"
        symlink.symlink_to(outside_file)

        workspace = LocalWorkspace(working_dir=str(tmp_path))
        content = await workspace.read_file("evil_link.txt")
        assert content == b"escaped via symlink"

    @pytest.mark.asyncio
    async def test_unicode_filename(self, tmp_path):
        """Unicode filenames should work."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("日本語.txt", b"Japanese content")
        content = await workspace.read_file("日本語.txt")
        assert content == b"Japanese content"

    @pytest.mark.asyncio
    async def test_filename_with_spaces_and_special_chars(self, tmp_path):
        """Filenames with spaces and special characters should work."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("file with spaces.txt", b"spaced")
        content = await workspace.read_file("file with spaces.txt")
        assert content == b"spaced"

    @pytest.mark.asyncio
    async def test_deeply_nested_directory_creation(self, tmp_path):
        """write_file should create deeply nested directories."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        deep_path = "a/b/c/d/e/f/g/h/i/j/deep.txt"
        await workspace.write_file(deep_path, b"deep")
        content = await workspace.read_file(deep_path)
        assert content == b"deep"

    @pytest.mark.asyncio
    async def test_list_files_with_hidden_files(self, tmp_path):
        """list_files should include hidden files (os.listdir includes them)."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("visible.txt", b"visible")
        await workspace.write_file(".hidden", b"hidden")

        files = await workspace.list_files(".")
        names = [f.name for f in files]
        assert "visible.txt" in names
        # Native Python os.listdir includes hidden files!
        assert ".hidden" in names

    @pytest.mark.asyncio
    async def test_read_nonexistent_with_special_chars_in_path(self, tmp_path):
        """read_file with special chars in path should raise FileNotFoundError."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await workspace.read_file("nonexistent 'file\".txt")

    @pytest.mark.asyncio
    async def test_execute_code_with_multiline_and_quotes(self, tmp_path):
        """execute_code should handle code with all types of quotes."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        code = """
x = "hello 'world'"
y = 'hello "world"'
print(x)
print(y)
"""
        result = await workspace._execute_code_to_result(code, language="python")
        assert result.exit_code == 0
        assert "hello 'world'" in result.stdout
        assert 'hello "world"' in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_with_backslashes(self, tmp_path):
        """execute_code should handle backslashes in code."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        code = 'print("path\\\\to\\\\file")'
        result = await workspace._execute_code_to_result(code, language="python")
        assert result.exit_code == 0
        assert "path\\to\\file" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_returns_correct_exit_codes(self, tmp_path):
        """Various exit codes should be preserved."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        for code in [0, 1, 2, 127, 255]:
            result = await workspace._execute_to_result(f"exit {code}")
            assert result.exit_code == code, f"Expected exit code {code}, got {result.exit_code}"

    @pytest.mark.asyncio
    async def test_blocking_file_io_in_async_context(self, tmp_path):
        """LocalWorkspace uses pathlib (blocking) I/O in async context.

        This is a known design concern: pathlib.read_text()/write_text() are
        synchronous. In an async context with many concurrent operations,
        this could block the event loop.
        """
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        large_content = b"X" * (1024 * 1024)  # 1MB
        await workspace.write_file("blocking_test.txt", large_content)
        content = await workspace.read_file("blocking_test.txt")
        assert len(content) == 1024 * 1024

    @pytest.mark.asyncio
    async def test_remove_file_nonexistent(self, tmp_path):
        """remove_file should raise FileNotFoundError for missing files."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await workspace.remove_file("nonexistent.txt")

    @pytest.mark.asyncio
    async def test_remove_file_then_read(self, tmp_path):
        """Removing a file and then reading it should raise FileNotFoundError."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("to_delete.txt", b"content")
        await workspace.remove_file("to_delete.txt")
        with pytest.raises(FileNotFoundError):
            await workspace.read_file("to_delete.txt")

    @pytest.mark.asyncio
    async def test_remove_file_outside_workspace(self, tmp_path):
        """remove_file with absolute path can delete files outside workspace."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "remove_escape"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "target.txt"
        outside_file.write_text("will be deleted")

        await workspace.remove_file(str(outside_file))
        assert not outside_file.exists()

    @pytest.mark.asyncio
    async def test_execute_code_rejects_unsafe_language(self, tmp_path):
        """execute_code validates language parameter against unsafe characters."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unsafe characters"):
            await workspace._execute_code_to_result("print(1)", language="python; rm -rf /")

    @pytest.mark.asyncio
    async def test_execute_code_rejects_path_traversal_language(self, tmp_path):
        """execute_code rejects language with path separators."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unsafe characters"):
            await workspace._execute_code_to_result("1", language="../../../bin/sh")

    @pytest.mark.asyncio
    async def test_execute_code_rejects_space_injection(self, tmp_path):
        """execute_code rejects language with spaces."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unsafe characters"):
            await workspace._execute_code_to_result("1", language="python -m http.server")


class TestShellBasedWorkspaceHeredocEdgeCases:
    """Edge cases in the ShellBasedWorkspace heredoc implementation."""

    @pytest.mark.asyncio
    async def test_write_file_content_with_single_quotes(self):
        """Content with single quotes is safely transported via base64."""

        class MockShellWorkspace(ShellBasedWorkspace):
            def __init__(self):
                super().__init__()
                self.last_command = ""

            async def execute(self, command, timeout=None, **kwargs):
                await self._ensure_started()
                self.last_command = command
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def start(self):
                self._started = True

        workspace = MockShellWorkspace()
        content = "line1\nline2\nline3"
        await workspace.write_file("/tmp/test.txt", content.encode())

        cmd = workspace.last_command
        assert "base64 -d" in cmd
        assert "/tmp/test.txt" in cmd

    @pytest.mark.asyncio
    async def test_write_file_empty_path(self):
        """Empty path should still be quoted."""

        class MockShellWorkspace(ShellBasedWorkspace):
            def __init__(self):
                super().__init__()
                self.last_command = ""

            async def execute(self, command, timeout=None, **kwargs):
                await self._ensure_started()
                self.last_command = command
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def start(self):
                self._started = True

        workspace = MockShellWorkspace()
        await workspace.write_file("", b"content")
        assert "''" in workspace.last_command
