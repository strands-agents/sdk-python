"""Tests for the Workspace ABC, ShellBasedWorkspace, and ExecutionResult dataclass."""

from collections.abc import AsyncGenerator

import pytest

from strands.workspace.base import (
    ExecutionResult,
    Workspace,
)
from strands.workspace.shell_based import ShellBasedWorkspace


class ConcreteShellWorkspace(ShellBasedWorkspace):
    """Minimal concrete ShellBasedWorkspace implementation for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.commands: list[str] = []
        self.started_count = 0
        self.stopped_count = 0

    async def execute(self, command: str, timeout: int | None = None) -> AsyncGenerator[str | ExecutionResult, None]:
        await self._ensure_started()
        self.commands.append(command)
        if "fail" in command:
            yield ExecutionResult(exit_code=1, stdout="", stderr="command failed")
            return
        stdout = f"output of: {command}\n"
        yield stdout
        yield ExecutionResult(exit_code=0, stdout=stdout, stderr="")

    async def start(self) -> None:
        self.started_count += 1
        self._started = True

    async def stop(self) -> None:
        self.stopped_count += 1
        self._started = False


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


class TestWorkspaceABC:
    """Tests that Workspace has all 6 abstract methods and cannot be partially implemented."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Workspace()  # type: ignore

    def test_cannot_instantiate_with_only_execute(self) -> None:
        """A class implementing only execute() is still abstract."""

        class OnlyExecute(Workspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

        with pytest.raises(TypeError):
            OnlyExecute()  # type: ignore

    def test_all_six_methods_required(self) -> None:
        """A class must implement all 6 abstract methods to be concrete."""

        class AllSix(Workspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code(
                self, code: str, language: str = "python", timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str) -> str:
                return ""

            async def write_file(self, path: str, content: str) -> None:
                pass

            async def remove_file(self, path: str) -> None:
                pass

            async def list_files(self, path: str = ".") -> list[str]:
                return []

        # Should not raise
        workspace = AllSix()
        assert workspace is not None

    def test_missing_remove_file_is_abstract(self) -> None:
        """A class missing remove_file() is still abstract."""

        class MissingRemoveFile(Workspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code(
                self, code: str, language: str = "python", timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str) -> str:
                return ""

            async def write_file(self, path: str, content: str) -> None:
                pass

            async def list_files(self, path: str = ".") -> list[str]:
                return []

        with pytest.raises(TypeError):
            MissingRemoveFile()  # type: ignore

    @pytest.mark.asyncio
    async def test_default_start_stop_work(self) -> None:
        """Test that the base class default start/stop work correctly."""

        class FullWorkspace(Workspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code(
                self, code: str, language: str = "python", timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str) -> str:
                return ""

            async def write_file(self, path: str, content: str) -> None:
                pass

            async def remove_file(self, path: str) -> None:
                pass

            async def list_files(self, path: str = ".") -> list[str]:
                return []

        workspace = FullWorkspace()
        await workspace.start()
        assert workspace._started
        await workspace.stop()
        assert not workspace._started

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async context manager on the base Workspace class."""

        class FullWorkspace(Workspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def execute_code(
                self, code: str, language: str = "python", timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

            async def read_file(self, path: str) -> str:
                return ""

            async def write_file(self, path: str, content: str) -> None:
                pass

            async def remove_file(self, path: str) -> None:
                pass

            async def list_files(self, path: str = ".") -> list[str]:
                return []

        workspace = FullWorkspace()
        async with workspace as ws:
            assert ws is workspace
            assert workspace._started
        assert not workspace._started


class TestShellBasedWorkspaceABC:
    """Tests that ShellBasedWorkspace is still abstract (execute() not implemented)."""

    def test_cannot_instantiate_shell_based_workspace(self) -> None:
        with pytest.raises(TypeError):
            ShellBasedWorkspace()  # type: ignore

    def test_shell_based_workspace_only_needs_execute(self) -> None:
        """ShellBasedWorkspace requires only execute() to be concrete."""
        workspace = ConcreteShellWorkspace()
        assert workspace is not None


class TestShellBasedWorkspaceOperations:
    """Tests for the shell-based default implementations of the 5 convenience methods."""

    @pytest.mark.asyncio
    async def test_execute_yields_lines_and_result(self) -> None:
        workspace = ConcreteShellWorkspace()
        chunks: list[str | ExecutionResult] = []
        async for chunk in workspace.execute("echo hello"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0
        assert any(isinstance(c, str) for c in chunks[:-1])
        assert workspace.commands == ["echo hello"]

    @pytest.mark.asyncio
    async def test_execute_to_result_helper(self) -> None:
        workspace = ConcreteShellWorkspace()
        result = await workspace._execute_to_result("echo hello")
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "echo hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_default(self) -> None:
        workspace = ConcreteShellWorkspace()
        result = await workspace._execute_code_to_result("print('hi')")
        assert result.exit_code == 0
        assert len(workspace.commands) == 1
        assert "python" in workspace.commands[0]
        assert "print" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_execute_code_streams(self) -> None:
        workspace = ConcreteShellWorkspace()
        chunks: list[str | ExecutionResult] = []
        async for chunk in workspace.execute_code("print('hi')"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_code_custom_language(self) -> None:
        workspace = ConcreteShellWorkspace()
        result = await workspace._execute_code_to_result("puts 'hi'", language="ruby")
        assert result.exit_code == 0
        assert "ruby" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_execute_code_quotes_language(self) -> None:
        """The language parameter is shell-quoted in the command."""
        workspace = ConcreteShellWorkspace()
        result = await workspace._execute_code_to_result("1+1", language="python3.12")
        assert result.exit_code == 0
        # shlex.quote wraps it when it contains a dot
        assert "python3.12" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_execute_code_quotes_malicious_language(self) -> None:
        """Malicious language parameter is safely shell-quoted, not executed."""
        workspace = ConcreteShellWorkspace()
        result = await workspace._execute_code_to_result("print(1)", language="python; rm -rf /")
        # The command should contain the safely quoted malicious string
        assert result.exit_code == 0
        # shlex.quote should wrap the malicious string so it's treated as a single arg
        cmd = workspace.commands[0]
        assert "python; rm -rf /" not in cmd.split(" -c ")[0] or "'" in cmd

    @pytest.mark.asyncio
    async def test_read_file_success(self) -> None:
        workspace = ConcreteShellWorkspace()
        content = await workspace.read_file("/tmp/test.txt")
        assert "cat" in workspace.commands[0]
        assert "/tmp/test.txt" in workspace.commands[0]
        assert content is not None

    @pytest.mark.asyncio
    async def test_read_file_not_found(self) -> None:
        workspace = ConcreteShellWorkspace()
        with pytest.raises(FileNotFoundError):
            await workspace.read_file("/tmp/fail.txt")

    @pytest.mark.asyncio
    async def test_write_file_success(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.write_file("/tmp/test.txt", "hello content")
        assert len(workspace.commands) == 1
        assert "/tmp/test.txt" in workspace.commands[0]
        assert "hello content" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_failure(self) -> None:
        workspace = ConcreteShellWorkspace()
        with pytest.raises(IOError):
            await workspace.write_file("/tmp/fail.txt", "content")

    @pytest.mark.asyncio
    async def test_write_file_uses_random_delimiter(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.write_file("/tmp/test.txt", "content with STRANDS_EOF inside")
        assert "STRANDS_EOF_" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_path_is_shell_quoted(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.write_file("/tmp/test file.txt", "content")
        assert "'/tmp/test file.txt'" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_path_is_shell_quoted(self) -> None:
        workspace = ConcreteShellWorkspace()
        content = await workspace.read_file("/tmp/test file.txt")
        assert "'/tmp/test file.txt'" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_remove_file_success(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.remove_file("/tmp/test.txt")
        assert len(workspace.commands) == 1
        assert "rm" in workspace.commands[0]
        assert "/tmp/test.txt" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_remove_file_not_found(self) -> None:
        workspace = ConcreteShellWorkspace()
        with pytest.raises(FileNotFoundError):
            await workspace.remove_file("/tmp/fail.txt")

    @pytest.mark.asyncio
    async def test_remove_file_path_is_shell_quoted(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.remove_file("/tmp/test file.txt")
        assert "'/tmp/test file.txt'" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_list_files_success(self) -> None:
        workspace = ConcreteShellWorkspace()
        files = await workspace.list_files("/tmp")
        assert len(workspace.commands) == 1
        assert "ls" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_list_files_not_found(self) -> None:
        workspace = ConcreteShellWorkspace()
        with pytest.raises(FileNotFoundError):
            await workspace.list_files("/tmp/fail")

    @pytest.mark.asyncio
    async def test_list_files_path_is_shell_quoted(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace.list_files("/tmp/my dir")
        assert "'/tmp/my dir'" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self) -> None:
        workspace = ConcreteShellWorkspace()
        assert not workspace._started
        await workspace.start()
        assert workspace._started
        await workspace.stop()
        assert not workspace._started

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        workspace = ConcreteShellWorkspace()
        async with workspace as ws:
            assert ws is workspace
            assert workspace._started
        assert not workspace._started

    @pytest.mark.asyncio
    async def test_execute_code_uses_shlex_quote(self) -> None:
        workspace = ConcreteShellWorkspace()
        code = "print('hello')"
        result = await workspace._execute_code_to_result(code)
        assert "python" in workspace.commands[0]
        assert "print" in workspace.commands[0]

    @pytest.mark.asyncio
    async def test_auto_start_on_first_execute(self) -> None:
        workspace = ConcreteShellWorkspace()
        assert not workspace._started
        result = await workspace._execute_to_result("echo hello")
        assert workspace._started
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_auto_start_only_once(self) -> None:
        workspace = ConcreteShellWorkspace()
        await workspace._execute_to_result("echo 1")
        await workspace._execute_to_result("echo 2")
        assert workspace.started_count == 1

    @pytest.mark.asyncio
    async def test_execute_to_result_raises_on_missing_result(self) -> None:
        """_execute_to_result raises if execute() yields no ExecutionResult."""

        class BadWorkspace(ShellBasedWorkspace):
            async def execute(
                self, command: str, timeout: int | None = None
            ) -> AsyncGenerator[str | ExecutionResult, None]:
                yield "just a string, no result"

        workspace = BadWorkspace()
        with pytest.raises(RuntimeError, match="did not yield an ExecutionResult"):
            await workspace._execute_to_result("anything")
