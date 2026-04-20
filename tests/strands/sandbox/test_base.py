"""Tests for the Sandbox ABC, ShellBasedSandbox, ExecutionResult, FileInfo, and OutputFile."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from strands.sandbox.base import (
    ExecutionResult,
    FileInfo,
    OutputFile,
    Sandbox,
    StreamChunk,
)
from strands.sandbox.shell_based import ShellBasedSandbox


class ConcreteShellSandbox(ShellBasedSandbox):
    """Minimal concrete ShellBasedSandbox implementation for testing."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute_streaming(
        self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        self.commands.append(command)
        if "fail" in command:
            yield ExecutionResult(exit_code=1, stdout="", stderr="command failed")
            return
        # For base64 commands (used by read_file), return valid base64 output
        if command.startswith("base64 "):
            import base64 as b64

            stdout = b64.b64encode(b"mock file content").decode("ascii") + "\n"
        else:
            stdout = f"output of: {command}\n"
        yield StreamChunk(data=stdout)
        yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")


class TestExecutionResult:
    def test_execution_result_fields(self) -> None:
        result = ExecutionResult(exit_code=0, stdout="hello", stderr="")
        assert result.exit_code == 0
        assert result.stdout == "hello"
        assert result.stderr == ""

    def test_execution_result_error(self) -> None:
        result = ExecutionResult(exit_code=1, stdout="", stderr="error msg")
        assert result.exit_code == 1
        assert result.stderr == "error msg"

    def test_execution_result_equality(self) -> None:
        r1 = ExecutionResult(exit_code=0, stdout="out", stderr="err")
        r2 = ExecutionResult(exit_code=0, stdout="out", stderr="err")
        assert r1 == r2

    def test_execution_result_output_files_default_empty(self) -> None:
        result = ExecutionResult(exit_code=0, stdout="", stderr="")
        assert result.output_files == []

    def test_execution_result_with_output_files(self) -> None:
        files = [OutputFile(name="plot.png", content=b"\x89PNG", mime_type="image/png")]
        result = ExecutionResult(exit_code=0, stdout="", stderr="", output_files=files)
        assert len(result.output_files) == 1
        assert result.output_files[0].name == "plot.png"
        assert result.output_files[0].content == b"\x89PNG"
        assert result.output_files[0].mime_type == "image/png"


class TestFileInfo:
    def test_file_info_file(self) -> None:
        info = FileInfo(name="test.txt", is_dir=False, size=1024)
        assert info.name == "test.txt"
        assert info.is_dir is False
        assert info.size == 1024

    def test_file_info_directory(self) -> None:
        info = FileInfo(name="subdir", is_dir=True)
        assert info.name == "subdir"
        assert info.is_dir is True
        assert info.size is None  # default is None now

    def test_file_info_equality(self) -> None:
        f1 = FileInfo(name="a.txt", is_dir=False, size=100)
        f2 = FileInfo(name="a.txt", is_dir=False, size=100)
        assert f1 == f2

    def test_file_info_optional_fields_default_none(self) -> None:
        """is_dir and size default to None when not provided."""
        info = FileInfo(name="unknown.txt")
        assert info.name == "unknown.txt"
        assert info.is_dir is None
        assert info.size is None

    def test_file_info_with_only_name(self) -> None:
        """FileInfo with only name is valid — unknown metadata."""
        info = FileInfo(name="mystery")
        assert info.is_dir is None
        assert info.size is None


class TestOutputFile:
    def test_output_file_fields(self) -> None:
        f = OutputFile(name="chart.svg", content=b"<svg></svg>", mime_type="image/svg+xml")
        assert f.name == "chart.svg"
        assert f.content == b"<svg></svg>"
        assert f.mime_type == "image/svg+xml"

    def test_output_file_default_mime_type(self) -> None:
        f = OutputFile(name="data.bin", content=b"\x00\x01")
        assert f.mime_type == "application/octet-stream"


class TestSandboxABC:
    """Tests that Sandbox has all abstract methods and cannot be partially implemented."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Sandbox()  # type: ignore

    def test_cannot_instantiate_with_only_execute_streaming(self) -> None:
        """A class implementing only execute_streaming() is still abstract."""

        class OnlyExecute(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

        with pytest.raises(TypeError):
            OnlyExecute()  # type: ignore

    def test_all_abstract_methods_required(self) -> None:
        """A class must implement all abstract methods to be concrete."""

        class AllMethods(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        # Should not raise
        sandbox = AllMethods()
        assert sandbox is not None

    def test_missing_remove_file_is_abstract(self) -> None:
        """A class missing remove_file() is still abstract."""

        class MissingRemoveFile(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        with pytest.raises(TypeError):
            MissingRemoveFile()  # type: ignore

    @pytest.mark.asyncio
    async def test_read_text_convenience(self) -> None:
        """Test that read_text decodes bytes from read_file."""

        class TextSandbox(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b"hello world"

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = TextSandbox()
        text = await sandbox.read_text("test.txt")
        assert text == "hello world"
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_write_text_convenience(self) -> None:
        """Test that write_text encodes string to bytes for write_file."""

        class TextSandbox(Sandbox):
            def __init__(self) -> None:
                self.written_content: bytes = b""

            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                self.written_content = content

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = TextSandbox()
        await sandbox.write_text("test.txt", "hello world")
        assert sandbox.written_content == b"hello world"

    @pytest.mark.asyncio
    async def test_non_streaming_execute_convenience(self) -> None:
        """Test that execute() returns ExecutionResult directly."""

        class SimpleSandbox(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield StreamChunk(data="output\n")
                yield ExecutionResult(exit_code=0, stdout="output\n", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = SimpleSandbox()
        result = await sandbox.execute("echo hello")
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout == "output\n"

    @pytest.mark.asyncio
    async def test_non_streaming_execute_code_convenience(self) -> None:
        """Test that execute_code() returns ExecutionResult directly."""

        class SimpleSandbox(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield StreamChunk(data="code output\n")
                yield ExecutionResult(exit_code=0, stdout="code output\n", stderr="")

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = SimpleSandbox()
        result = await sandbox.execute_code("print(1)", language="python")
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert result.stdout == "code output\n"

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_result(self) -> None:
        """execute() raises RuntimeError if execute_streaming yields no ExecutionResult."""

        class BadSandbox(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield "just a string"

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield "just a string"

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = BadSandbox()
        with pytest.raises(RuntimeError, match="did not yield an ExecutionResult"):
            await sandbox.execute("anything")

    @pytest.mark.asyncio
    async def test_execute_code_raises_on_missing_result(self) -> None:
        """execute_code() raises RuntimeError if execute_code_streaming yields no ExecutionResult."""

        class BadSandbox(Sandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield "just a string"

            async def execute_code_streaming(
                self, code: str, language: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                yield "just a string"

            async def read_file(self, path: str, **kwargs: Any) -> bytes:
                return b""

            async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
                pass

            async def remove_file(self, path: str, **kwargs: Any) -> None:
                pass

            async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
                return []

        sandbox = BadSandbox()
        with pytest.raises(RuntimeError, match="did not yield an ExecutionResult"):
            await sandbox.execute_code("print(1)", language="python")


class TestShellBasedSandboxABC:
    """Tests that ShellBasedSandbox is still abstract (execute_streaming() not implemented)."""

    def test_cannot_instantiate_shell_based_sandbox(self) -> None:
        with pytest.raises(TypeError):
            ShellBasedSandbox()  # type: ignore

    def test_shell_based_sandbox_only_needs_execute_streaming(self) -> None:
        """ShellBasedSandbox requires only execute_streaming() to be concrete."""
        sandbox = ConcreteShellSandbox()
        assert sandbox is not None


class TestShellBasedSandboxOperations:
    """Tests for the shell-based default implementations."""

    @pytest.mark.asyncio
    async def test_execute_streaming_yields_lines_and_result(self) -> None:
        sandbox = ConcreteShellSandbox()
        chunks: list[StreamChunk | ExecutionResult] = []
        async for chunk in sandbox.execute_streaming("echo hello"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0
        assert any(isinstance(c, StreamChunk) for c in chunks[:-1])
        assert sandbox.commands == ["echo hello"]

    @pytest.mark.asyncio
    async def test_non_streaming_execute(self) -> None:
        sandbox = ConcreteShellSandbox()
        result = await sandbox.execute("echo hello")
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "echo hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_streaming_default(self) -> None:
        sandbox = ConcreteShellSandbox()
        chunks: list[StreamChunk | ExecutionResult] = []
        async for chunk in sandbox.execute_code_streaming("print('hi')", language="python"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0

    @pytest.mark.asyncio
    async def test_non_streaming_execute_code(self) -> None:
        sandbox = ConcreteShellSandbox()
        result = await sandbox.execute_code("print('hi')", language="python")
        assert result.exit_code == 0
        assert len(sandbox.commands) == 1
        assert "python" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_execute_code_custom_language(self) -> None:
        sandbox = ConcreteShellSandbox()
        result = await sandbox.execute_code("puts 'hi'", language="ruby")
        assert result.exit_code == 0
        assert "ruby" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_returns_bytes(self) -> None:
        sandbox = ConcreteShellSandbox()
        content = await sandbox.read_file("/tmp/test.txt")
        assert isinstance(content, bytes)
        assert content == b"mock file content"
        assert "base64" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_not_found(self) -> None:
        sandbox = ConcreteShellSandbox()
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("/tmp/fail.txt")

    @pytest.mark.asyncio
    async def test_write_file_accepts_bytes(self) -> None:
        sandbox = ConcreteShellSandbox()
        await sandbox.write_file("/tmp/test.txt", b"hello content")
        assert len(sandbox.commands) == 1
        assert "/tmp/test.txt" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_failure(self) -> None:
        sandbox = ConcreteShellSandbox()
        with pytest.raises(IOError):
            await sandbox.write_file("/tmp/fail.txt", b"content")

    @pytest.mark.asyncio
    async def test_write_file_uses_base64(self) -> None:
        sandbox = ConcreteShellSandbox()
        await sandbox.write_file("/tmp/test.txt", b"content with STRANDS_EOF inside")
        assert "base64 -d" in sandbox.commands[0]
        assert "/tmp/test.txt" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_path_is_shell_quoted(self) -> None:
        sandbox = ConcreteShellSandbox()
        await sandbox.write_file("/tmp/test file.txt", b"content")
        assert "'/tmp/test file.txt'" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_path_is_shell_quoted(self) -> None:
        sandbox = ConcreteShellSandbox()
        await sandbox.read_file("/tmp/test file.txt")
        assert "'/tmp/test file.txt'" in sandbox.commands[0]
        assert "base64" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_remove_file_success(self) -> None:
        sandbox = ConcreteShellSandbox()
        await sandbox.remove_file("/tmp/test.txt")
        assert len(sandbox.commands) == 1
        assert "rm" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_remove_file_not_found(self) -> None:
        sandbox = ConcreteShellSandbox()
        with pytest.raises(FileNotFoundError):
            await sandbox.remove_file("/tmp/fail.txt")

    @pytest.mark.asyncio
    async def test_list_files_returns_file_info(self) -> None:
        sandbox = ConcreteShellSandbox()
        files = await sandbox.list_files("/tmp")
        assert len(sandbox.commands) == 1
        assert "ls" in sandbox.commands[0]
        # The mock returns a string output, so list_files parses it
        for f in files:
            assert isinstance(f, FileInfo)

    @pytest.mark.asyncio
    async def test_list_files_not_found(self) -> None:
        sandbox = ConcreteShellSandbox()
        with pytest.raises(FileNotFoundError):
            await sandbox.list_files("/tmp/fail")

    @pytest.mark.asyncio
    async def test_list_files_size_is_none(self) -> None:
        """ShellBasedSandbox.list_files returns size=None (cannot determine from ls)."""
        sandbox = ConcreteShellSandbox()
        files = await sandbox.list_files("/tmp")
        for f in files:
            assert f.size is None


class TestShellBasedListFilesRealisticOutput:
    @pytest.mark.asyncio
    async def test_list_files_parses_directory_indicator(self) -> None:
        class RealisticLsSandbox(ShellBasedSandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                stdout = ".\n..\nsubdir/\n.hidden_dir/\nfile.txt\nscript.py*\nlink.txt@\npipe_file|\n"
                yield StreamChunk(data=stdout)
                yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")

        sandbox = RealisticLsSandbox()
        files = await sandbox.list_files("/some/path")

        names = [f.name for f in files]
        is_dir_map = {f.name: f.is_dir for f in files}

        assert "subdir" in names
        assert is_dir_map["subdir"] is True
        assert ".hidden_dir" in names
        assert is_dir_map[".hidden_dir"] is True
        assert "file.txt" in names
        assert is_dir_map["file.txt"] is False
        assert "script.py" in names
        assert is_dir_map["script.py"] is False
        assert "link.txt" in names
        assert is_dir_map["link.txt"] is False
        assert "pipe_file" in names
        assert is_dir_map["pipe_file"] is False
        assert "." not in names
        assert ".." not in names

    @pytest.mark.asyncio
    async def test_list_files_empty_directory(self) -> None:
        class EmptyLsSandbox(ShellBasedSandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                stdout = "./\n../\n"
                yield StreamChunk(data=stdout)
                yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")

        sandbox = EmptyLsSandbox()
        files = await sandbox.list_files("/empty")
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_files_only_no_indicators(self) -> None:
        class PlainLsSandbox(ShellBasedSandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                stdout = "readme.md\nsetup.py\nrequirements.txt\n"
                yield StreamChunk(data=stdout)
                yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")

        sandbox = PlainLsSandbox()
        files = await sandbox.list_files("/project")
        names = [f.name for f in files]
        assert names == ["readme.md", "setup.py", "requirements.txt"]
        assert all(not f.is_dir for f in files)
        assert all(f.size is None for f in files)  # shell-based has no size info


class TestShellBasedWriteFileParentDirs:
    @pytest.mark.asyncio
    async def test_write_file_command_includes_mkdir(self) -> None:
        class CommandCapture(ShellBasedSandbox):
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.commands.append(command)
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

        sandbox = CommandCapture()
        await sandbox.write_file("/tmp/deep/nested/file.txt", b"content")
        assert len(sandbox.commands) == 1
        assert "mkdir -p" in sandbox.commands[0]
        assert "base64 -d" in sandbox.commands[0]


class TestShellBasedBase64Encoding:
    @pytest.mark.asyncio
    async def test_write_file_encodes_exact_base64(self) -> None:
        import base64 as b64

        class CommandCapture(ShellBasedSandbox):
            def __init__(self) -> None:
                self.commands: list[str] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.commands.append(command)
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

        sandbox = CommandCapture()
        original_content = b"hello world with special chars: \x00\xff\n\t"
        await sandbox.write_file("/tmp/test.bin", original_content)
        expected_b64 = b64.b64encode(original_content).decode("ascii")
        assert expected_b64 in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_decodes_base64_correctly(self) -> None:
        import base64 as b64

        original_content = b"\x89PNG\r\n\x1a\n\x00\x00binary"

        class Base64Sandbox(ShellBasedSandbox):
            async def execute_streaming(
                self, command: str, timeout: int | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                stdout = b64.b64encode(original_content).decode("ascii") + "\n"
                yield StreamChunk(data=stdout)
                yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")

        sandbox = Base64Sandbox()
        content = await sandbox.read_file("/tmp/image.png")
        assert content == original_content


class TestShellBasedSandboxCwdPassthrough:
    """Test that ShellBasedSandbox passes cwd through to execute_streaming."""

    @pytest.mark.asyncio
    async def test_execute_code_streaming_passes_cwd(self) -> None:
        """execute_code_streaming should forward cwd to execute_streaming."""

        class CwdTrackingSandbox(ShellBasedSandbox):
            def __init__(self) -> None:
                self.received_cwds: list[str | None] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.received_cwds.append(cwd)
                yield ExecutionResult(exit_code=0, stdout="ok", stderr="")

        sandbox = CwdTrackingSandbox()
        await sandbox.execute_code("print(1)", language="python", cwd="/custom/dir")
        assert sandbox.received_cwds == ["/custom/dir"]

    @pytest.mark.asyncio
    async def test_execute_code_streaming_passes_none_cwd_by_default(self) -> None:
        """When no cwd is provided, None should be passed through."""

        class CwdTrackingSandbox(ShellBasedSandbox):
            def __init__(self) -> None:
                self.received_cwds: list[str | None] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.received_cwds.append(cwd)
                yield ExecutionResult(exit_code=0, stdout="ok", stderr="")

        sandbox = CwdTrackingSandbox()
        await sandbox.execute_code("print(1)", language="python")
        assert sandbox.received_cwds == [None]


class TestNonStreamingConveniencePassesCwd:
    """Test that non-streaming execute/execute_code pass cwd to streaming methods."""

    @pytest.mark.asyncio
    async def test_execute_passes_cwd_to_streaming(self) -> None:
        """The non-streaming execute() should forward cwd to execute_streaming()."""

        class CwdTrackingSandbox(ShellBasedSandbox):
            def __init__(self) -> None:
                self.received_cwds: list[str | None] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.received_cwds.append(cwd)
                yield ExecutionResult(exit_code=0, stdout="ok", stderr="")

        sandbox = CwdTrackingSandbox()
        result = await sandbox.execute("echo hi", cwd="/some/path")
        assert sandbox.received_cwds == ["/some/path"]
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_code_passes_cwd_to_streaming(self) -> None:
        """The non-streaming execute_code() should forward cwd to execute_code_streaming()."""

        class CwdTrackingSandbox(ShellBasedSandbox):
            def __init__(self) -> None:
                self.received_cwds: list[str | None] = []

            async def execute_streaming(
                self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs: Any
            ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
                self.received_cwds.append(cwd)
                yield ExecutionResult(exit_code=0, stdout="ok", stderr="")

        sandbox = CwdTrackingSandbox()
        result = await sandbox.execute_code("print(1)", language="python", cwd="/code/dir")
        assert sandbox.received_cwds == ["/code/dir"]
        assert result.exit_code == 0
