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
from typing import Any

from .base import ExecutionResult, FileInfo, Workspace

logger = logging.getLogger(__name__)

#: Maximum number of bytes to read at once from a subprocess stream.
#: Prevents memory exhaustion from extremely long lines without newlines.
_READ_CHUNK_SIZE = 64 * 1024  # 64 KiB

#: Pattern for validating language/interpreter names.
#: Allows alphanumeric characters, dots, hyphens, and underscores.
_LANGUAGE_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


async def _read_stream(
    stream: asyncio.StreamReader | None,
    collected: list[str],
) -> None:
    """Read all chunks from a subprocess stream into a list.

    Reads in chunks of up to ``_READ_CHUNK_SIZE`` bytes to handle
    binary output and extremely long lines without newlines.

    Args:
        stream: The subprocess stdout or stderr stream.
        collected: List to append decoded string chunks to.
    """
    if stream is None:
        return
    while True:
        chunk_bytes = await stream.read(_READ_CHUNK_SIZE)
        if not chunk_bytes:
            break
        collected.append(chunk_bytes.decode())


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
            raise NotADirectoryError(f"working_dir is not a directory: {self.working_dir}")
        working_path.mkdir(parents=True, exist_ok=True)
        logger.debug("working_dir=<%s> | local workspace started", self.working_dir)
        await super().start()

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute a shell command on the local host, streaming output.

        Reads stdout and stderr in chunks (up to 64 KiB at a time) to avoid
        blocking on extremely long lines. Chunks are collected and yielded
        after the command completes. The final yield is an ExecutionResult
        with the exit code and complete captured output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. None means no timeout.
            **kwargs: Additional keyword arguments for forward compatibility.

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

        async for item in self._collect_and_yield(proc, timeout):
            yield item

    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code on the local host using subprocess_exec (no shell intermediary).

        Uses :func:`asyncio.create_subprocess_exec` to invoke the language
        interpreter directly, passing code via the ``-c`` flag. This avoids
        shell quoting issues entirely — the interpreter name and code are
        passed as separate arguments to ``execvp``, not concatenated into a
        shell command string.

        The language parameter is validated against a safe pattern to prevent
        path traversal or binary injection. If the interpreter is not found
        on the system, an ExecutionResult with exit code 127 is returned
        (matching the shell convention for "command not found").

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use (e.g.
                ``"python"``, ``"python3"``, ``"node"``, ``"ruby"``).
            timeout: Maximum execution time in seconds. None means no timeout.
            **kwargs: Additional keyword arguments for forward compatibility.

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
            raise ValueError(f"language parameter contains unsafe characters: {language}")

        logger.debug(
            "language=<%s>, timeout=<%s> | executing code locally",
            language,
            timeout,
        )

        # Use create_subprocess_exec (not shell) — the interpreter and arguments
        # are passed directly to execvp, avoiding all shell quoting issues.
        try:
            proc = await asyncio.create_subprocess_exec(
                language,
                "-c",
                code,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            yield ExecutionResult(
                exit_code=127,
                stdout="",
                stderr=f"Language interpreter not found: {language}",
            )
            return

        async for item in self._collect_and_yield(proc, timeout):
            yield item

    async def _collect_and_yield(
        self,
        proc: asyncio.subprocess.Process,
        timeout: int | None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Read stdout/stderr from a subprocess, then yield chunks and a final ExecutionResult.

        Shared helper used by both ``execute()`` and ``execute_code()`` to
        avoid duplicating the stream-reading, timeout-handling, and yielding logic.

        Args:
            proc: The running subprocess.
            timeout: Maximum time in seconds to wait for output. None means no timeout.

        Yields:
            str chunks of output, then a final ExecutionResult.

        Raises:
            asyncio.TimeoutError: If the process exceeds the timeout.
        """
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

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

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the local filesystem as raw bytes.

        Uses ``asyncio.to_thread`` to avoid blocking the event loop
        during disk I/O.

        Args:
            path: Path to the file to read. Relative paths are resolved
                against the working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._resolve_path(path)
        return await asyncio.to_thread(full_path.read_bytes)

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write bytes to a file on the local filesystem.

        Creates parent directories if they do not exist. Uses
        ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            path: Path to the file to write. Relative paths are resolved
                against the working directory.
            content: The content to write as bytes.
            **kwargs: Additional keyword arguments for forward compatibility.
        """
        full_path = self._resolve_path(path)

        def _write() -> None:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(content)

        await asyncio.to_thread(_write)

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the local filesystem using native Python methods.

        Uses ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            path: Path to the file to remove. Relative paths are resolved
                against the working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full_path = self._resolve_path(path)
        await asyncio.to_thread(full_path.unlink)

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a directory with structured metadata.

        Uses native Python methods (:func:`os.listdir`, :func:`os.stat`)
        to return :class:`FileInfo` entries with name, is_dir, and size.
        Results include hidden files (dotfiles) and are sorted for
        deterministic ordering.

        Uses ``asyncio.to_thread`` to avoid blocking the event loop
        during disk I/O.

        Args:
            path: Path to the directory to list. Relative paths are resolved
                against the working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A sorted list of :class:`FileInfo` entries.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        full_path = self._resolve_path(path)

        def _list() -> list[FileInfo]:
            if not full_path.is_dir():
                raise FileNotFoundError(f"Directory not found: {full_path}")
            entries = []
            for name in sorted(os.listdir(full_path)):
                entry_path = full_path / name
                try:
                    stat = entry_path.stat()
                    entries.append(
                        FileInfo(
                            name=name,
                            is_dir=entry_path.is_dir(),
                            size=stat.st_size,
                        )
                    )
                except OSError:
                    # If we can't stat the entry (e.g., broken symlink), include
                    # it with defaults
                    entries.append(FileInfo(name=name, is_dir=False, size=0))
            return entries

        return await asyncio.to_thread(_list)
