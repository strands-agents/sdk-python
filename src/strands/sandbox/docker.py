"""Docker sandbox implementation for containerized execution.

This module implements the DockerSandbox, which executes commands and code
inside a Docker container. The container is created on start() and destroyed
on stop(). Each execute() call uses ``docker exec`` on the running container.

Docker must be available on the host and the user must have permission to run
containers.
"""

import asyncio
import logging
import shlex
from collections.abc import AsyncGenerator
from typing import Any

from .base import ExecutionResult, ShellBasedSandbox

logger = logging.getLogger(__name__)


class DockerSandbox(ShellBasedSandbox):
    """Execute code and commands in a Docker container.

    The container is created during start() and removed during stop().
    Commands run via ``docker exec`` on the running container, so filesystem
    state persists across execute() calls for the lifetime of the container.
    Working directory and environment variables set via ``export`` do not
    carry across calls (each ``docker exec`` starts a new shell process).

    Args:
        image: Docker image to use for the container.
        volumes: Host-to-container volume mounts as ``{host_path: container_path}``.
        environment: Environment variables to set in the container.
        working_dir: Working directory inside the container.
        docker_command: Path to the docker CLI binary.

    Example:
        ```python
        from strands.sandbox.docker import DockerSandbox

        async with DockerSandbox(image="python:3.12-slim") as sandbox:
            async for chunk in sandbox.execute("python -c 'print(1+1)'"):
                if isinstance(chunk, str):
                    print(chunk, end="")
        ```
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        volumes: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        working_dir: str = "/workspace",
        docker_command: str = "docker",
    ) -> None:
        """Initialize the DockerSandbox.

        Args:
            image: Docker image to use for the container.
            volumes: Host-to-container volume mounts as ``{host_path: container_path}``.
            environment: Environment variables to set in the container.
            working_dir: Working directory inside the container.
            docker_command: Path to the docker CLI binary.
        """
        super().__init__()
        self.image = image
        self.volumes = volumes or {}
        self.environment = environment or {}
        self.working_dir = working_dir
        self.docker_command = docker_command
        self._container_id: str | None = None

    async def _run_docker(
        self,
        args: list[str],
        timeout: int | None = None,
        stdin_data: bytes | None = None,
    ) -> ExecutionResult:
        """Run a docker CLI command and return the result.

        This is a low-level helper used by lifecycle methods (start/stop)
        and write_file. It does NOT stream — it collects all output at once.

        Args:
            args: Arguments to pass to the docker command.
            timeout: Maximum execution time in seconds.
            stdin_data: Optional data to send to stdin.

        Returns:
            The result of the docker command.

        Raises:
            asyncio.TimeoutError: If the command exceeds the timeout.
        """
        cmd = [self.docker_command] + args
        logger.debug("docker_args=<%s> | running docker command", " ".join(args))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        return ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create and start the Docker container.

        Raises:
            RuntimeError: If the container cannot be created.
        """
        if self._container_id is not None:
            self._started = True
            return

        create_args = ["create", "--rm", "-i"]

        # Working directory
        create_args += ["-w", self.working_dir]

        # Volume mounts
        for host_path, container_path in self.volumes.items():
            create_args += ["-v", f"{host_path}:{container_path}"]

        # Environment variables
        for key, value in self.environment.items():
            create_args += ["-e", f"{key}={value}"]

        create_args.append(self.image)
        # Keep the container alive with a long-running sleep
        create_args += ["sleep", "infinity"]

        result = await self._run_docker(create_args, timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(f"failed to create docker container: {result.stderr}")

        self._container_id = result.stdout.strip()
        logger.debug("container_id=<%s> | created docker container", self._container_id)

        # Start the container
        start_result = await self._run_docker(["start", self._container_id], timeout=30)
        if start_result.exit_code != 0:
            raise RuntimeError(f"failed to start docker container: {start_result.stderr}")

        self._started = True

        # Ensure working directory exists
        await self._execute_to_result(f"mkdir -p {shlex.quote(self.working_dir)}")

        logger.info("container_id=<%s>, image=<%s> | docker sandbox started", self._container_id, self.image)

    async def stop(self) -> None:
        """Stop and remove the Docker container."""
        if self._container_id is None:
            self._started = False
            return

        container_id = self._container_id
        self._container_id = None
        self._started = False

        try:
            await self._run_docker(["rm", "-f", container_id], timeout=30)
            logger.info("container_id=<%s> | docker sandbox stopped", container_id)
        except Exception as e:
            logger.warning("container_id=<%s>, error=<%s> | failed to remove container", container_id, e)

    async def __aenter__(self) -> "DockerSandbox":
        """Enter the async context manager, starting the sandbox."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager, stopping the sandbox."""
        await self.stop()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute a shell command inside the Docker container, streaming output.

        Reads stdout and stderr line by line from the ``docker exec`` process
        and yields each line. The final yield is an ExecutionResult.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds.

        Yields:
            str lines of output, then a final ExecutionResult.

        Raises:
            RuntimeError: If the sandbox has not been started.
            asyncio.TimeoutError: If the command exceeds the timeout.
        """
        await self._ensure_started()
        if self._container_id is None:
            raise RuntimeError("docker sandbox has not been started, call start() or use as async context manager")

        exec_args = [
            self.docker_command,
            "exec",
            "-w",
            self.working_dir,
            self._container_id,
            "sh",
            "-c",
            command,
        ]

        logger.debug("docker_exec=<%s> | executing in container", command)

        proc = await asyncio.create_subprocess_exec(
            *exec_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def _read_stream(stream: asyncio.StreamReader | None, collected: list[str]) -> None:
            if stream is None:
                return
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                collected.append(line_bytes.decode())

        try:
            read_task = asyncio.gather(
                _read_stream(proc.stdout, stdout_lines),
                _read_stream(proc.stderr, stderr_lines),
            )
            await asyncio.wait_for(read_task, timeout=timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)

        for line in stdout_lines:
            yield line
        for line in stderr_lines:
            yield line

        yield ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    # ------------------------------------------------------------------
    # File I/O overrides (use stdin pipe for reliability)
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> None:
        """Write a file into the container by piping content via stdin.

        Uses ``docker exec`` with stdin to avoid heredoc injection issues.
        Content is piped directly to ``cat`` inside the container, so any
        file content (including shell metacharacters) is handled safely.

        Args:
            path: Path inside the container. Relative paths are resolved
                against the working directory.
            content: The content to write.

        Raises:
            RuntimeError: If the sandbox has not been started.
            IOError: If the file cannot be written.
        """
        await self._ensure_started()
        if self._container_id is None:
            raise RuntimeError("docker sandbox has not been started")

        # Resolve relative paths
        if not path.startswith("/"):
            path = f"{self.working_dir}/{path}"

        # Ensure parent directory exists
        parent = "/".join(path.split("/")[:-1])
        if parent:
            await self._execute_to_result(f"mkdir -p {shlex.quote(parent)}")

        # Pipe content via stdin to avoid heredoc injection
        exec_args = [
            "exec",
            "-i",
            "-w",
            self.working_dir,
            self._container_id,
            "sh",
            "-c",
            f"cat > {shlex.quote(path)}",
        ]
        result = await self._run_docker(exec_args, stdin_data=content.encode())
        if result.exit_code != 0:
            raise IOError(result.stderr)

    async def read_file(self, path: str) -> str:
        """Read a file from the container.

        Args:
            path: Path inside the container. Relative paths are resolved
                against the working directory.

        Returns:
            The file contents as a string.

        Raises:
            RuntimeError: If the sandbox has not been started.
            FileNotFoundError: If the file does not exist.
        """
        await self._ensure_started()
        if self._container_id is None:
            raise RuntimeError("docker sandbox has not been started")

        if not path.startswith("/"):
            path = f"{self.working_dir}/{path}"

        result = await self._execute_to_result(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)
        return result.stdout
