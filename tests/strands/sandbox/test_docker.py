"""Tests for the DockerSandbox implementation."""

import asyncio
import unittest.mock

import pytest

from strands.sandbox.base import ExecutionResult, ShellBasedSandbox
from strands.sandbox.docker import DockerSandbox


@pytest.fixture
def sandbox() -> DockerSandbox:
    """Create a DockerSandbox with a fake container ID for unit tests."""
    s = DockerSandbox(image="python:3.12-slim", working_dir="/workspace")
    s._container_id = "fake-container-123"
    s._started = True
    return s


class TestDockerSandboxInit:
    def test_defaults(self) -> None:
        s = DockerSandbox()
        assert s.image == "python:3.12-slim"
        assert s.working_dir == "/workspace"
        assert s.volumes == {}
        assert s.environment == {}
        assert s._container_id is None

    def test_custom_params(self) -> None:
        s = DockerSandbox(
            image="node:20",
            volumes={"/host": "/container"},
            environment={"FOO": "bar"},
            working_dir="/app",
        )
        assert s.image == "node:20"
        assert s.volumes == {"/host": "/container"}
        assert s.environment == {"FOO": "bar"}
        assert s.working_dir == "/app"

    def test_inherits_shell_based_sandbox(self) -> None:
        s = DockerSandbox()
        assert isinstance(s, ShellBasedSandbox)


class TestDockerSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_not_started_raises(self) -> None:
        s = DockerSandbox()
        # _ensure_started will call start(), which will fail because docker isn't available
        # We mock start to simply set _started=True but leave _container_id as None
        async def mock_start() -> None:
            s._started = True

        with unittest.mock.patch.object(s, "start", side_effect=mock_start):
            with pytest.raises(RuntimeError, match="has not been started"):
                async for _ in s.execute("echo hello"):
                    pass

    @pytest.mark.asyncio
    async def test_execute_yields_lines_and_result(self, sandbox: DockerSandbox) -> None:
        """execute() streams lines and yields a final ExecutionResult."""

        async def mock_create_subprocess_exec(*args: object, **kwargs: object) -> unittest.mock.AsyncMock:
            proc = unittest.mock.AsyncMock()
            proc.returncode = 0
            proc.stdout = _make_stream_reader(b"hello\nworld\n")
            proc.stderr = _make_stream_reader(b"")
            proc.wait = unittest.mock.AsyncMock(return_value=0)
            return proc

        with unittest.mock.patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
            chunks: list[str | ExecutionResult] = []
            async for chunk in sandbox.execute("echo hello"):
                chunks.append(chunk)

        str_chunks = [c for c in chunks if isinstance(c, str)]
        result_chunks = [c for c in chunks if isinstance(c, ExecutionResult)]
        assert len(result_chunks) == 1
        assert result_chunks[0].exit_code == 0
        assert result_chunks[0].stdout == "hello\nworld\n"
        assert "hello\n" in str_chunks
        assert "world\n" in str_chunks

    @pytest.mark.asyncio
    async def test_execute_returns_exit_code(self, sandbox: DockerSandbox) -> None:
        async def mock_create_subprocess_exec(*args: object, **kwargs: object) -> unittest.mock.AsyncMock:
            proc = unittest.mock.AsyncMock()
            proc.returncode = 42
            proc.stdout = _make_stream_reader(b"")
            proc.stderr = _make_stream_reader(b"bad command\n")
            proc.wait = unittest.mock.AsyncMock(return_value=42)
            return proc

        with unittest.mock.patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
            result = await sandbox._execute_to_result("bad_cmd")

        assert result.exit_code == 42
        assert "bad command" in result.stderr


class TestDockerSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_and_starts_container(self) -> None:
        s = DockerSandbox(image="python:3.12-slim", volumes={"/host": "/cont"}, environment={"A": "1"})

        call_count = 0

        async def mock_run_docker(
            args: list[str], timeout: int | None = None, stdin_data: bytes | None = None
        ) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if args[0] == "create":
                assert "-v" in args
                assert "-e" in args
                return ExecutionResult(exit_code=0, stdout="container-abc123\n", stderr="")
            elif args[0] == "start":
                return ExecutionResult(exit_code=0, stdout="", stderr="")
            return ExecutionResult(exit_code=1, stdout="", stderr="unexpected")

        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        with unittest.mock.patch.object(s, "_run_docker", side_effect=mock_run_docker):
            with unittest.mock.patch.object(s, "_execute_to_result", side_effect=mock_execute_to_result):
                await s.start()

        assert s._container_id == "container-abc123"
        assert s._started
        assert call_count >= 2  # create + start

    @pytest.mark.asyncio
    async def test_start_raises_on_create_failure(self) -> None:
        s = DockerSandbox()

        async def mock_run_docker(
            args: list[str], timeout: int | None = None, stdin_data: bytes | None = None
        ) -> ExecutionResult:
            return ExecutionResult(exit_code=1, stdout="", stderr="no such image")

        with unittest.mock.patch.object(s, "_run_docker", side_effect=mock_run_docker):
            with pytest.raises(RuntimeError, match="failed to create"):
                await s.start()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, sandbox: DockerSandbox) -> None:
        """start() is a no-op if container already exists."""
        with unittest.mock.patch.object(sandbox, "_run_docker") as mock:
            await sandbox.start()
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_removes_container(self, sandbox: DockerSandbox) -> None:
        mock_result = ExecutionResult(exit_code=0, stdout="", stderr="")
        with unittest.mock.patch.object(sandbox, "_run_docker", return_value=mock_result) as mock_run:
            await sandbox.stop()

        mock_run.assert_called_once_with(["rm", "-f", "fake-container-123"], timeout=30)
        assert sandbox._container_id is None
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_stop_noop_if_not_started(self) -> None:
        s = DockerSandbox()
        await s.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        s = DockerSandbox()
        start_calls: list[bool] = []
        stop_calls: list[bool] = []

        async def mock_start() -> None:
            s._container_id = "ctx-container"
            s._started = True
            start_calls.append(True)

        async def mock_stop() -> None:
            s._container_id = None
            s._started = False
            stop_calls.append(True)

        with unittest.mock.patch.object(s, "start", side_effect=mock_start):
            with unittest.mock.patch.object(s, "stop", side_effect=mock_stop):
                async with s as ctx:
                    assert ctx is s
                    assert len(start_calls) == 1

        assert len(stop_calls) == 1


class TestDockerSandboxFileOps:
    @pytest.mark.asyncio
    async def test_write_file_not_started(self) -> None:
        s = DockerSandbox()
        # Mock start to not actually start docker
        async def mock_start() -> None:
            s._started = True

        with unittest.mock.patch.object(s, "start", side_effect=mock_start):
            with pytest.raises(RuntimeError, match="has not been started"):
                await s.write_file("test.txt", "content")

    @pytest.mark.asyncio
    async def test_read_file_not_started(self) -> None:
        s = DockerSandbox()

        async def mock_start() -> None:
            s._started = True

        with unittest.mock.patch.object(s, "start", side_effect=mock_start):
            with pytest.raises(RuntimeError, match="has not been started"):
                await s.read_file("test.txt")

    @pytest.mark.asyncio
    async def test_write_file_relative_path(self, sandbox: DockerSandbox) -> None:
        """write_file resolves relative paths, creates dirs, and pipes via stdin."""
        calls: list[tuple[str, ...]] = []

        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            calls.append(("execute_to_result", command))
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        async def mock_run_docker(
            args: list[str], timeout: int | None = None, stdin_data: bytes | None = None
        ) -> ExecutionResult:
            calls.append(("run_docker", str(args), str(stdin_data)))
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        with unittest.mock.patch.object(sandbox, "_execute_to_result", side_effect=mock_execute_to_result):
            with unittest.mock.patch.object(sandbox, "_run_docker", side_effect=mock_run_docker):
                await sandbox.write_file("data/test.txt", "hello")

        assert any("mkdir" in str(c) for c in calls)
        assert any("hello" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_write_file_uses_stdin_pipe(self, sandbox: DockerSandbox) -> None:
        """Verify write_file uses stdin piping instead of heredoc."""
        calls: list[dict[str, object]] = []

        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        async def mock_run_docker(
            args: list[str], timeout: int | None = None, stdin_data: bytes | None = None
        ) -> ExecutionResult:
            calls.append({"args": args, "stdin_data": stdin_data})
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        with unittest.mock.patch.object(sandbox, "_execute_to_result", side_effect=mock_execute_to_result):
            with unittest.mock.patch.object(sandbox, "_run_docker", side_effect=mock_run_docker):
                await sandbox.write_file("test.txt", "content with STRANDS_EOF inside")

        write_calls = [c for c in calls if c["stdin_data"] is not None]
        assert len(write_calls) == 1
        assert write_calls[0]["stdin_data"] == b"content with STRANDS_EOF inside"
        assert "-i" in write_calls[0]["args"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_read_file_success(self, sandbox: DockerSandbox) -> None:
        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            return ExecutionResult(exit_code=0, stdout="file content", stderr="")

        with unittest.mock.patch.object(sandbox, "_execute_to_result", side_effect=mock_execute_to_result):
            content = await sandbox.read_file("test.txt")
        assert content == "file content"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, sandbox: DockerSandbox) -> None:
        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            return ExecutionResult(exit_code=1, stdout="", stderr="No such file")

        with unittest.mock.patch.object(sandbox, "_execute_to_result", side_effect=mock_execute_to_result):
            with pytest.raises(FileNotFoundError):
                await sandbox.read_file("missing.txt")

    @pytest.mark.asyncio
    async def test_write_file_io_error(self, sandbox: DockerSandbox) -> None:
        async def mock_execute_to_result(command: str, timeout: int | None = None) -> ExecutionResult:
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        async def mock_run_docker(
            args: list[str], timeout: int | None = None, stdin_data: bytes | None = None
        ) -> ExecutionResult:
            if stdin_data is not None:
                return ExecutionResult(exit_code=1, stdout="", stderr="permission denied")
            return ExecutionResult(exit_code=0, stdout="", stderr="")

        with unittest.mock.patch.object(sandbox, "_execute_to_result", side_effect=mock_execute_to_result):
            with unittest.mock.patch.object(sandbox, "_run_docker", side_effect=mock_run_docker):
                with pytest.raises(IOError):
                    await sandbox.write_file("readonly/test.txt", "content")


class TestDockerSandboxExecuteCode:
    @pytest.mark.asyncio
    async def test_execute_code_streams(self, sandbox: DockerSandbox) -> None:
        """execute_code uses the ShellBasedSandbox default and streams."""

        async def mock_create_subprocess_exec(*args: object, **kwargs: object) -> unittest.mock.AsyncMock:
            proc = unittest.mock.AsyncMock()
            proc.returncode = 0
            proc.stdout = _make_stream_reader(b"42\n")
            proc.stderr = _make_stream_reader(b"")
            proc.wait = unittest.mock.AsyncMock(return_value=0)
            return proc

        with unittest.mock.patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
            chunks: list[str | ExecutionResult] = []
            async for chunk in sandbox.execute_code("print(42)"):
                chunks.append(chunk)

        assert isinstance(chunks[-1], ExecutionResult)
        assert chunks[-1].stdout == "42\n"


def _make_stream_reader(data: bytes) -> asyncio.StreamReader:
    """Create an asyncio.StreamReader pre-loaded with data."""
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader
