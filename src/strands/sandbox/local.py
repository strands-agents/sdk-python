"""Local sandbox implementation for host-process execution.

This module implements the LocalSandbox, which executes commands and code
on the local host using asyncio subprocesses. It overrides read_file and
write_file with native filesystem calls for encoding safety.

This is the default sandbox used when no explicit sandbox is configured.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from .base import ExecutionResult, ShellBasedSandbox

logger = logging.getLogger(__name__)

#: Maximum number of bytes to read at once from a subprocess stream.
#: Prevents memory exhaustion from extremely long lines without newlines.
_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB


class LocalSandbox(ShellBasedSandbox):
    """Execute code and commands on the local host.

    Uses asyncio subprocesses for command execution and native filesystem
    operations for file I/O. This is the default sandbox, providing the
    same behavior as running commands directly on the host.

    Args:
        working_dir: The working directory for command execution.
            Defaults to the current working directory.

    Example:
        ```python
        from strands.sandbox import LocalSandbox

        sandbox = LocalSandbox(working_dir="/tmp/workspace")
        async for chunk in sandbox.execute("echo hello"):
            if isinstance(chunk, str):
                print(chunk, end="")
        ```
    """

    def __init__(self, working_dir: str | None = None) -> None:
        """Initialize the LocalSandbox.

        Args:
            working_dir: The working directory for command execution.
                Defaults to the current working directory at construction time.
        """
        super().__init__()
        self.working_dir = working_dir or os.getcwd()

    async def start(self) -> None:
        """Initialize the sandbox and ensure the working directory exists.

        Creates the working directory if it does not exist. Raises a clear
        error if the path exists but is not a directory.

        Raises:
            NotADirectoryError: If the working_dir path exists but is not a directory.
        """
        working_path = Path(self.working_dir)
        if working_path.exists() and not working_path.is_dir():
            raise NotADirectoryError(
                "working_dir is not a directory: %s" % self.working_dir
            )
        working_path.mkdir(parents=True, exist_ok=True)
        logger.debug("working_dir=<%s> | local sandbox started", self.working_dir)
        self._started = True

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute a shell command on the local host, streaming output.

        Reads stdout and stderr in chunks (up to 64 KiB at a time) to avoid
        blocking on extremely long lines. Each chunk is yielded as it
        arrives. The final yield is an ExecutionResult with the exit code
        and complete captured output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. None means no timeout.

        Yields:
            str chunks of output, then a final ExecutionResult.

        Raises:
            asyncio.TimeoutError: If the command exceeds the timeout.
        """
        await self._ensure_started()
        logger.debug("command=<%s>, timeout=<%s> | executing local command", command, timeout)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _read_stream(
            stream: asyncio.StreamReader | None,
            collected: list[str],
        ) -> None:
            if stream is None:
                return
            while True:
                chunk_bytes = await stream.read(_READ_CHUNK_SIZE)
                if not chunk_bytes:
                    break
                collected.append(chunk_bytes.decode())

        try:
            read_task = asyncio.gather(
                _read_stream(proc.stdout, stdout_chunks),
                _read_stream(proc.stderr, stderr_chunks),
            )
            await asyncio.wait_for(read_task, timeout=timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        # Yield each collected chunk as a streaming piece
        for chunk in stdout_chunks:
            yield chunk
        for chunk in stderr_chunks:
            yield chunk

        # Final yield: the complete ExecutionResult
        yield ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    async def read_file(self, path: str) -> str:
        """Read a file from the local filesystem.

        Uses native file I/O instead of shell commands for encoding safety.

        Args:
            path: Path to the file to read. Relative paths are resolved
                against the working directory.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
        with open(full_path) as f:
            return f.read()

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the local filesystem.

        Uses native file I/O instead of shell commands for encoding safety.

        Args:
            path: Path to the file to write. Relative paths are resolved
                against the working directory.
            content: The content to write to the file.
        """
        full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
        parent_dir = os.path.dirname(full_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
