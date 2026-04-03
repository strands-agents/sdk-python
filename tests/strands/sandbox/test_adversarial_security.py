"""Adversarial tests: Security, path traversal, injection, and edge cases.

Tests for:
- Path traversal attacks in LocalSandbox
- Heredoc delimiter collision in ShellBasedSandbox
- Language validation bypasses
- Symlink attacks
- Binary/special content handling
- Docker readline hanging on no-newline output
"""

import asyncio
import os
import unittest.mock

import pytest

from strands.sandbox.base import (
    ALLOWED_LANGUAGES,
    ExecutionResult,
    ShellBasedSandbox,
    _validate_language,
)
from strands.sandbox.local import LocalSandbox
from strands.sandbox.docker import DockerSandbox


class TestPathTraversal:
    """Can we escape the working directory via path traversal?"""

    @pytest.mark.asyncio
    async def test_read_file_path_traversal(self, tmp_path):
        """read_file with ../.. should still work (LocalSandbox uses native I/O)."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # Create a file outside the working dir
        outside_dir = tmp_path.parent / "outside_sandbox"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("SECRET_DATA")

        # LocalSandbox doesn't restrict path traversal — it uses os.path.join
        # which joins the relative path to working_dir but doesn't prevent ..
        relative_path = f"../outside_sandbox/secret.txt"
        content = await sandbox.read_file(relative_path)
        # BUG: LocalSandbox allows path traversal outside working_dir
        # This WILL succeed — proving LocalSandbox doesn't sandbox file access
        assert content == "SECRET_DATA", "Path traversal should be documented or blocked"

    @pytest.mark.asyncio
    async def test_write_file_path_traversal(self, tmp_path):
        """write_file with ../.. can write outside working directory."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        outside_dir = tmp_path.parent / "write_escape"
        outside_dir.mkdir(exist_ok=True)

        relative_path = "../write_escape/pwned.txt"
        await sandbox.write_file(relative_path, "PWNED")

        # Verify the file was written outside the sandbox
        assert (outside_dir / "pwned.txt").read_text() == "PWNED"

    @pytest.mark.asyncio
    async def test_execute_can_access_entire_filesystem(self, tmp_path):
        """execute() runs arbitrary shell commands — it can read /etc/passwd."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        result = await sandbox._execute_to_result("cat /etc/passwd | head -1")
        assert result.exit_code == 0
        assert "root" in result.stdout  # LocalSandbox has no filesystem isolation

    @pytest.mark.asyncio
    async def test_absolute_path_bypasses_working_dir(self, tmp_path):
        """Absolute paths completely bypass working_dir for LocalSandbox."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        # Write outside via absolute path
        abs_path = str(tmp_path.parent / "abs_escape.txt")
        await sandbox.write_file(abs_path, "escaped")
        content = await sandbox.read_file(abs_path)
        assert content == "escaped"


class TestHeredocInjection:
    """Can content that matches the delimiter break write_file?"""

    @pytest.mark.asyncio
    async def test_content_containing_strands_eof(self, tmp_path):
        """Content containing STRANDS_EOF_ prefix should not break heredoc."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # This content tries to end the heredoc early
        malicious_content = "line1\nSTRANDS_EOF_deadbeef\nline3"
        await sandbox.write_file("test.txt", malicious_content)
        content = await sandbox.read_file("test.txt")
        assert content == malicious_content, f"Heredoc injection: got {content!r}"

    @pytest.mark.asyncio
    async def test_content_with_shell_metacharacters(self, tmp_path):
        """Content with shell metacharacters should be preserved."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = 'hello $USER `whoami` $(id) && rm -rf / ; echo pwned'
        await sandbox.write_file("test.txt", content)
        read_back = await sandbox.read_file("test.txt")
        assert read_back == content

    @pytest.mark.asyncio
    async def test_content_with_null_bytes(self, tmp_path):
        """Content with null bytes should be handled or raise a clear error."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = "before\x00after"
        # Native file I/O can handle null bytes in text mode
        await sandbox.write_file("null.txt", content)
        read_back = await sandbox.read_file("null.txt")
        assert read_back == content

    @pytest.mark.asyncio
    async def test_empty_content(self, tmp_path):
        """Writing empty content should create an empty file."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("empty.txt", "")
        content = await sandbox.read_file("empty.txt")
        assert content == ""

    @pytest.mark.asyncio
    async def test_very_large_content(self, tmp_path):
        """Writing 10MB of content should work."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        large_content = "A" * (10 * 1024 * 1024)  # 10 MB
        await sandbox.write_file("large.txt", large_content)
        content = await sandbox.read_file("large.txt")
        assert len(content) == len(large_content)
        assert content == large_content


class TestLanguageValidationBypass:
    """Can we bypass language validation?"""

    def test_path_separator_in_language(self):
        """Language with path separator should be rejected."""
        with pytest.raises(ValueError):
            _validate_language("/usr/bin/python")

    def test_language_with_equals(self):
        """Language with = should be rejected (env var injection)."""
        with pytest.raises(ValueError):
            _validate_language("FOO=bar python")

    def test_language_with_ampersand(self):
        """Language with & should be rejected."""
        with pytest.raises(ValueError):
            _validate_language("python&rm")

    def test_language_with_newline(self):
        """Language with newline should be rejected."""
        with pytest.raises(ValueError):
            _validate_language("python\nrm -rf /")

    def test_language_dot_dot_slash(self):
        """Language like ../../evil should be rejected."""
        with pytest.raises(ValueError):
            _validate_language("../../evil")

    def test_language_rm(self):
        """Language 'rm' passes validation — it's alphanumeric. Is this intentional?"""
        # This PASSES _validate_language because it matches _LANGUAGE_RE
        # But 'rm -c <code>' would try to run rm with -c flag
        # The code is also shlex.quote'd, so it's 'rm' -c '<code>'
        # rm doesn't have -c flag so it would just error out
        _validate_language("rm")  # This should not raise

    def test_language_chmod(self):
        """Language 'chmod' passes validation — alphanumeric."""
        _validate_language("chmod")  # This should not raise

    def test_language_with_long_name(self):
        """Very long language name should be accepted (no length limit)."""
        _validate_language("a" * 10000)  # No max length check


class TestDockerSandboxReadlineHang:
    """DockerSandbox uses readline() which hangs on output without newlines."""

    @pytest.mark.asyncio
    async def test_docker_execute_no_newline_output(self):
        """DockerSandbox.execute() uses readline() — hangs if output has no newlines.
        
        LocalSandbox uses read() with a chunk size, but DockerSandbox uses readline().
        If a command outputs 1MB without a newline, readline() will buffer the entire
        thing in memory and only return when it hits EOF or a newline.
        
        This is a design inconsistency between LocalSandbox and DockerSandbox.
        """
        sandbox = DockerSandbox()
        sandbox._container_id = "fake-container"
        sandbox._started = True

        # Simulate a process that outputs 1MB without newlines
        huge_output = b"A" * (1024 * 1024)  # 1MB, no newline

        async def mock_create_subprocess_exec(*args, **kwargs):
            proc = unittest.mock.AsyncMock()
            proc.returncode = 0
            reader = asyncio.StreamReader()
            reader.feed_data(huge_output)
            reader.feed_eof()
            proc.stdout = reader
            proc.stderr = asyncio.StreamReader()
            proc.stderr.feed_eof()
            proc.wait = unittest.mock.AsyncMock(return_value=0)
            return proc

        with unittest.mock.patch(
            "asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec
        ):
            chunks = []
            async for chunk in sandbox.execute("cat /dev/urandom"):
                chunks.append(chunk)

        # DockerSandbox uses readline() — this will read the entire 1MB as one line
        # because there are no newlines. The readline() call buffers everything until EOF.
        str_chunks = [c for c in chunks if isinstance(c, str)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert len(result_chunks[0].stdout) == 1024 * 1024
        # This works but demonstrates the inconsistency:
        # LocalSandbox yields 64KB chunks, DockerSandbox yields the entire output as one chunk


class TestLocalSandboxEdgeCases:
    """Edge cases specific to LocalSandbox."""

    @pytest.mark.asyncio
    async def test_symlink_read(self, tmp_path):
        """Reading through a symlink should work."""
        real_file = tmp_path / "real.txt"
        real_file.write_text("real content")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = await sandbox.read_file("link.txt")
        assert content == "real content"

    @pytest.mark.asyncio
    async def test_symlink_outside_sandbox(self, tmp_path):
        """Symlink pointing outside working_dir — LocalSandbox follows it."""
        outside_dir = tmp_path.parent / "symlink_escape"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "target.txt"
        outside_file.write_text("escaped via symlink")

        symlink = tmp_path / "evil_link.txt"
        symlink.symlink_to(outside_file)

        sandbox = LocalSandbox(working_dir=str(tmp_path))
        content = await sandbox.read_file("evil_link.txt")
        # BUG: Symlink escape — reads file outside sandbox
        assert content == "escaped via symlink"

    @pytest.mark.asyncio
    async def test_unicode_filename(self, tmp_path):
        """Unicode filenames should work."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("日本語.txt", "Japanese content")
        content = await sandbox.read_file("日本語.txt")
        assert content == "Japanese content"

    @pytest.mark.asyncio
    async def test_filename_with_spaces_and_special_chars(self, tmp_path):
        """Filenames with spaces and special characters should work."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("file with spaces.txt", "spaced")
        content = await sandbox.read_file("file with spaces.txt")
        assert content == "spaced"

    @pytest.mark.asyncio
    async def test_deeply_nested_directory_creation(self, tmp_path):
        """write_file should create deeply nested directories."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        deep_path = "a/b/c/d/e/f/g/h/i/j/deep.txt"
        await sandbox.write_file(deep_path, "deep")
        content = await sandbox.read_file(deep_path)
        assert content == "deep"

    @pytest.mark.asyncio
    async def test_list_files_with_hidden_files(self, tmp_path):
        """list_files should include hidden files (ls -1 shows them by default? No, it doesn't)."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("visible.txt", "visible")
        await sandbox.write_file(".hidden", "hidden")

        files = await sandbox.list_files(".")
        # ls -1 does NOT show hidden files by default
        assert "visible.txt" in files
        # BUG: Hidden files are invisible to list_files because it uses `ls -1`
        # which doesn't include hidden files. Should it use `ls -1a` instead?
        assert ".hidden" not in files  # This is a design limitation

    @pytest.mark.asyncio
    async def test_read_nonexistent_with_special_chars_in_path(self, tmp_path):
        """read_file with special chars in path should raise FileNotFoundError."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("nonexistent 'file\".txt")

    @pytest.mark.asyncio
    async def test_execute_code_with_multiline_and_quotes(self, tmp_path):
        """execute_code should handle code with all types of quotes."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = """
x = "hello 'world'"
y = 'hello "world"'
print(x)
print(y)
"""
        result = await sandbox._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "hello 'world'" in result.stdout
        assert 'hello "world"' in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_with_backslashes(self, tmp_path):
        """execute_code should handle backslashes in code."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        code = r'print("path\\to\\file")'
        result = await sandbox._execute_code_to_result(code)
        assert result.exit_code == 0
        assert "path\\to\\file" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_returns_correct_exit_codes(self, tmp_path):
        """Various exit codes should be preserved."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        for code in [0, 1, 2, 127, 255]:
            result = await sandbox._execute_to_result(f"exit {code}")
            assert result.exit_code == code, f"Expected exit code {code}, got {result.exit_code}"

    @pytest.mark.asyncio
    async def test_blocking_file_io_in_async_context(self, tmp_path):
        """LocalSandbox.read_file/write_file use BLOCKING file I/O in async context.
        
        This is a design concern: open() and f.read()/f.write() are synchronous
        blocking calls. In an async context with many concurrent operations,
        this could block the event loop.
        """
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        # Write a large file synchronously inside async
        large_content = "X" * (1024 * 1024)  # 1MB
        await sandbox.write_file("blocking_test.txt", large_content)
        content = await sandbox.read_file("blocking_test.txt")
        assert len(content) == 1024 * 1024
        # This works but blocks the event loop during I/O


class TestShellBasedSandboxHeredocEdgeCases:
    """Edge cases in the ShellBasedSandbox heredoc implementation."""

    @pytest.mark.asyncio
    async def test_write_file_content_with_single_quotes(self):
        """Content with single quotes used in heredoc delimiter quoting."""
        # The heredoc uses: cat > path << 'DELIMITER'
        # Single quotes around delimiter prevent variable expansion
        # But what if content itself manipulates the heredoc?

        class MockShellSandbox(ShellBasedSandbox):
            def __init__(self):
                super().__init__()
                self.last_command = ""

            async def execute(self, command, timeout=None):
                await self._ensure_started()
                self.last_command = command
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def start(self):
                self._started = True

        sandbox = MockShellSandbox()
        content = "line1\nline2\nline3"
        await sandbox.write_file("/tmp/test.txt", content)

        # Verify the heredoc command is well-formed
        cmd = sandbox.last_command
        assert "STRANDS_EOF_" in cmd
        # shlex.quote only quotes paths with shell metacharacters
        # /tmp/test.txt is safe, so no extra quoting is applied
        assert "/tmp/test.txt" in cmd
        assert content in cmd

    @pytest.mark.asyncio
    async def test_write_file_empty_path(self):
        """Empty path should still be quoted."""

        class MockShellSandbox(ShellBasedSandbox):
            def __init__(self):
                super().__init__()
                self.last_command = ""

            async def execute(self, command, timeout=None):
                await self._ensure_started()
                self.last_command = command
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def start(self):
                self._started = True

        sandbox = MockShellSandbox()
        await sandbox.write_file("", "content")
        # Empty path gets quoted as '' by shlex.quote
        assert "''" in sandbox.last_command
