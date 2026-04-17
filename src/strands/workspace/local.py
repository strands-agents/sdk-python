"""Local workspace implementation for host-process execution.

This module implements the LocalWorkspace, which executes commands and code
on the local host using asyncio subprocesses and native Python filesystem
operations. It extends Workspace directly — all file and code operations
use proper Python methods (pathlib, os, subprocess) instead of shell commands.

This is the default workspace used when no explicit workspace is configured.
"""

import asyncio
import logging
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path

from .base import ExecutionResult, Workspace

logger = logging.getLogger(__name__)

#: Maximum number of bytes to read at once from a subprocess stream.
#: Prevents memory exhaustion from extremely long lines without newlines.
_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB

#: Pattern for validating language/interpreter names.
#: Allows alphanumeric characters, dots, hyphens, and underscores.
_LANGUAGE_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


class LocalWorkspace(Workspace):
    """Execute code and commands on the local host using native Python methods.

    Uses asyncio subprocesses for command execution, ``subprocess_exec`` for
    code execution (avoiding shell intermediaries), and native filesystem
    operations (``pathlib``, ``os``) for all file I/O.

    This workspace extends :class:`Workspace` directly — it does **not**
    inherit from :class:`ShellBasedWorkspace`. All operations use proper,
    safe Python methods instead of piping through shell commands.

    Args:
        working_dir: The working directory for command execution.
            Defaults to the current working directory.

    Example:
        ```python
        from strands.workspace import LocalWorkspace

        workspace = LocalWorkspace(working_dir="/tmp/my-workspace")
        async for chunk in workspace.execute("echo hello"):
            if isinstance(chunk, str):
                print(chunk, end="")
        ```
    """

    def __init__(self, working_dir: str | None = None) -> None:
        """Initialize the LocalWorkspace.

        Args:
            working_dir: The working directory for command execution.
                Defaults to the current working directory at construction time.
        """
        super().__init__()
        self.working_dir = working_dir or os.getcwd()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the working directory.

        Absolute paths are returned as-is. Relative paths are resolved
        against the working directory.

        Args:
            path: The file path to resolve.

        Returns:
            The resolved Path object.
        """
        if os.path.isabs(path):
            return Path(path)
        return Path(self.working_dir) / path

    async def start(self) -> None:
        """Initialize the workspace and ensure the working directory exists.

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
        logger.debug("working_dir=<%s> | local workspace started", self.working_dir)
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
            exit_code=0 if proc.returncode is None else proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code on the local host using subprocess_exec (no shell intermediary).

        Uses :func:`asyncio.create_subprocess_exec` to invoke the language
        interpreter directly, passing code via the ``-c`` flag. This avoids
        shell quoting issues entirely — the interpreter name and code are
        passed as separate arguments to ``execvp``, not concatenated into a
        shell command string.

        The language parameter is validated against a safe pattern to prevent
        path traversal or binary injection.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use (e.g.
                ``"python"``, ``"python3"``, ``"node"``, ``"ruby"``).
            timeout: Maximum execution time in seconds. None means no timeout.

        Yields:
            str chunks of output, then a final ExecutionResult.

        Raises:
            asyncio.TimeoutError: If the code execution exceeds the timeout.
            ValueError: If the language parameter contains unsafe characters.
        """
        await self._ensure_started()

        # Validate language to prevent injection via interpreter name.
        # Only allow safe characters (alphanumeric, dots, hyphens, underscores).
        if not _LANGUAGE_PATTERN.match(language):
            raise ValueError(
                "language parameter contains unsafe characters: %s" % language
            )

        logger.debug(
            "language=<%s>, timeout=<%s> | executing code locally",
            language,
            timeout,
        )

        # Use create_subprocess_exec (not shell) — the interpreter and arguments
        # are passed directly to execvp, avoiding all shell quoting issues.
        proc = await asyncio.create_subprocess_exec(
            language,
            "-c",
            code,
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

        for chunk in stdout_chunks:
            yield chunk
        for chunk in stderr_chunks:
            yield chunk

        yield ExecutionResult(
            exit_code=0 if proc.returncode is None else proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    async def read_file(self, path: str) -> str:
        """Read a file from the local filesystem using native Python I/O.

        Args:
            path: Path to the file to read. Relative paths are resolved
                against the working directory.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._resolve_path(path)
        return full_path.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the local filesystem using native Python I/O.

        Creates parent directories if they do not exist.

        Args:
            path: Path to the file to write. Relative paths are resolved
                against the working directory.
            content: The content to write to the file.
        """
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    async def remove_file(self, path: str) -> None:
        """Remove a file from the local filesystem using native Python methods.

        Args:
            path: Path to the file to remove. Relative paths are resolved
                against the working directory.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._resolve_path(path)
        full_path.unlink()

    async def list_files(self, path: str = ".") -> list[str]:
        """List files in a directory using native Python methods.

        Uses :func:`os.listdir` which includes hidden files (dotfiles),
        unlike the shell ``ls -1`` command. Results are sorted for
        deterministic ordering.

        Args:
            path: Path to the directory to list. Relative paths are resolved
                against the working directory.

        Returns:
            A sorted list of filenames in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        full_path = self._resolve_path(path)
        if not full_path.is_dir():
            raise FileNotFoundError("Directory not found: %s" % full_path)
        return sorted(os.listdir(full_path))
