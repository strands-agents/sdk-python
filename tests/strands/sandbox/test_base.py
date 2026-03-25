"""Tests for the Sandbox ABC and ExecutionResult dataclass."""

from collections.abc import AsyncGenerator

import pytest

from strands.sandbox.base import ExecutionResult, Sandbox


class ConcreteSandbox(Sandbox):
    """Minimal concrete implementation for testing the ABC."""

    def __init__(self):
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
    def test_execution_result_fields(self):
        result = ExecutionResult(exit_code=0, stdout="hello", stderr="")
        assert result.exit_code == 0
        assert result.stdout == "hello"
        assert result.stderr == ""

    def test_execution_result_error(self):
        result = ExecutionResult(exit_code=1, stdout="", stderr="error msg")
        assert result.exit_code == 1
        assert result.stderr == "error msg"

    def test_execution_result_equality(self):
        r1 = ExecutionResult(exit_code=0, stdout="out", stderr="err")
        r2 = ExecutionResult(exit_code=0, stdout="out", stderr="err")
        assert r1 == r2


class TestSandboxABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Sandbox()  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_yields_lines_and_result(self):
        sandbox = ConcreteSandbox()
        chunks = []
        async for chunk in sandbox.execute("echo hello"):
            chunks.append(chunk)
        # Last chunk is ExecutionResult
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0
        # Earlier chunks are strings
        assert any(isinstance(c, str) for c in chunks[:-1])
        assert sandbox.commands == ["echo hello"]

    @pytest.mark.asyncio
    async def test_execute_to_result_helper(self):
        sandbox = ConcreteSandbox()
        result = await sandbox._execute_to_result("echo hello")
        assert isinstance(result, ExecutionResult)
        assert result.exit_code == 0
        assert "echo hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_default(self):
        sandbox = ConcreteSandbox()
        result = await sandbox._execute_code_to_result("print('hi')")
        assert result.exit_code == 0
        # Default implementation pipes code through shell via shlex.quote
        assert len(sandbox.commands) == 1
        assert "python" in sandbox.commands[0]
        assert "print" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_execute_code_streams(self):
        sandbox = ConcreteSandbox()
        chunks = []
        async for chunk in sandbox.execute_code("print('hi')"):
            chunks.append(chunk)
        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].exit_code == 0

    @pytest.mark.asyncio
    async def test_execute_code_custom_language(self):
        sandbox = ConcreteSandbox()
        result = await sandbox._execute_code_to_result("puts 'hi'", language="ruby")
        assert result.exit_code == 0
        assert "ruby" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        sandbox = ConcreteSandbox()
        content = await sandbox.read_file("/tmp/test.txt")
        assert "cat" in sandbox.commands[0]
        assert "/tmp/test.txt" in sandbox.commands[0]
        assert content is not None

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        sandbox = ConcreteSandbox()
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("/tmp/fail.txt")

    @pytest.mark.asyncio
    async def test_write_file_success(self):
        sandbox = ConcreteSandbox()
        await sandbox.write_file("/tmp/test.txt", "hello content")
        assert len(sandbox.commands) == 1
        assert "/tmp/test.txt" in sandbox.commands[0]
        assert "hello content" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_failure(self):
        sandbox = ConcreteSandbox()
        with pytest.raises(IOError):
            await sandbox.write_file("/tmp/fail.txt", "content")

    @pytest.mark.asyncio
    async def test_write_file_uses_random_delimiter(self):
        sandbox = ConcreteSandbox()
        await sandbox.write_file("/tmp/test.txt", "content with STRANDS_EOF inside")
        assert "STRANDS_EOF_" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_write_file_path_is_shell_quoted(self):
        sandbox = ConcreteSandbox()
        await sandbox.write_file("/tmp/test file.txt", "content")
        assert "'/tmp/test file.txt'" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_read_file_path_is_shell_quoted(self):
        sandbox = ConcreteSandbox()
        content = await sandbox.read_file("/tmp/test file.txt")
        assert "'/tmp/test file.txt'" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_list_files_success(self):
        sandbox = ConcreteSandbox()
        files = await sandbox.list_files("/tmp")
        assert len(sandbox.commands) == 1
        assert "ls" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_list_files_not_found(self):
        sandbox = ConcreteSandbox()
        with pytest.raises(FileNotFoundError):
            await sandbox.list_files("/tmp/fail")

    @pytest.mark.asyncio
    async def test_list_files_path_is_shell_quoted(self):
        sandbox = ConcreteSandbox()
        await sandbox.list_files("/tmp/my dir")
        assert "'/tmp/my dir'" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self):
        sandbox = ConcreteSandbox()
        assert not sandbox._started

        await sandbox.start()
        assert sandbox._started

        await sandbox.stop()
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        sandbox = ConcreteSandbox()
        async with sandbox as s:
            assert s is sandbox
            assert sandbox._started
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_default_start_stop_are_noop(self):
        """Test that the base class default start/stop work correctly."""

        class MinimalSandbox(Sandbox):
            async def execute(self, command: str, timeout: int | None = None) -> AsyncGenerator[str | ExecutionResult, None]:
                yield ExecutionResult(exit_code=0, stdout="", stderr="")

        sandbox = MinimalSandbox()
        await sandbox.start()
        assert sandbox._started
        await sandbox.stop()
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_execute_code_uses_shlex_quote(self):
        sandbox = ConcreteSandbox()
        code = "print('hello')"
        result = await sandbox._execute_code_to_result(code)
        assert "python" in sandbox.commands[0]
        assert "print" in sandbox.commands[0]

    @pytest.mark.asyncio
    async def test_auto_start_on_first_execute(self):
        sandbox = ConcreteSandbox()
        assert not sandbox._started
        result = await sandbox._execute_to_result("echo hello")
        assert sandbox._started
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_auto_start_only_once(self):
        sandbox = ConcreteSandbox()
        await sandbox._execute_to_result("echo 1")
        await sandbox._execute_to_result("echo 2")
        assert sandbox.started_count == 1

    @pytest.mark.asyncio
    async def test_execute_to_result_raises_on_missing_result(self):
        """_execute_to_result raises if execute() yields no ExecutionResult."""

        class BadSandbox(Sandbox):
            async def execute(self, command: str, timeout: int | None = None) -> AsyncGenerator[str | ExecutionResult, None]:
                yield "just a string, no result"

        sandbox = BadSandbox()
        with pytest.raises(RuntimeError, match="did not yield an ExecutionResult"):
            await sandbox._execute_to_result("anything")
