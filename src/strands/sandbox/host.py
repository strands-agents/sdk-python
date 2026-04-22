"""Host sandbox implementation for host-process execution.

This module implements the HostSandbox, which executes commands and code
on the local host using asyncio subprocesses and native Python filesystem
operations. It extends Sandbox directly — all file and code operations
use proper Python methods (pathlib, os, subprocess) instead of shell commands.

This is the default sandbox used when no explicit sandbox is configured.
"""

import asyncio
import logging
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from .base import ExecutionResult, FileInfo, Sandbox, StreamChunk

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
    Non-UTF-8 bytes are replaced with the Unicode replacement character
    to prevent ``UnicodeDecodeError`` from crashing the sandbox.

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
        collected.append(chunk_bytes.decode(errors="replace"))


class HostSandbox(Sandbox):
    """Execute code and commands on the local host using native Python methods.

    Uses asyncio subprocesses for command execution, ``subprocess_exec`` for
    code execution (avoiding shell intermediaries), and native filesystem
    operations (``pathlib``, ``os``) for all file I/O.

    This sandbox extends :class:`Sandbox` directly — it does **not**
    inherit from :class:`ShellBasedSandbox`. All operations use proper,
    safe Python methods instead of piping through shell commands.

    Args:
        working_dir: The working directory for command execution.
            Defaults to the current working directory.

    Example:
        Non-streaming (common case)::

            from strands.sandbox import HostSandbox

            sandbox = HostSandbox(working_dir="/tmp/my-sandbox")
            result = await sandbox.execute("echo hello")
            print(result.stdout)

        Streaming::

            async for chunk in sandbox.execute_streaming("echo hello"):
                if isinstance(chunk, StreamChunk):
                    print(chunk.data, end="")
    """

    def __init__(self, working_dir: str | None = None) -> None:
        """Initialize the HostSandbox.

        Args:
            working_dir: The working directory for command execution.
                Defaults to the current working directory at construction time.
        """
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

    async def execute_streaming(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a shell command on the local host, streaming output.

        Reads stdout and stderr in chunks (up to 64 KiB at a time) to avoid
        blocking on extremely long lines. Chunks are collected and yielded
        after the command completes. The final yield is an ExecutionResult
        with the exit code and complete captured output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for this command. ``None`` means use the
                sandbox's default ``working_dir``.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for stdout/stderr, then a final :class:`ExecutionResult`.

        Raises:
            asyncio.TimeoutError: If the command exceeds the timeout.
        """
        effective_cwd = cwd or self.working_dir
        logger.debug("command=<%s>, timeout=<%s>, cwd=<%s> | executing local command", command, timeout, effective_cwd)

        working_path = Path(effective_cwd)
        working_path.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=effective_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async for item in self._collect_and_yield(proc, timeout):
            yield item

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
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
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for code execution. ``None`` means use the
                sandbox's default ``working_dir``.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for stdout/stderr, then a final :class:`ExecutionResult`.

        Raises:
            asyncio.TimeoutError: If the code execution exceeds the timeout.
            ValueError: If the language parameter contains unsafe characters.
        """
        # Validate language to prevent injection via interpreter name.
        # Only allow safe characters (alphanumeric, dots, hyphens, underscores).
        if not _LANGUAGE_PATTERN.match(language):
            raise ValueError(f"language parameter contains unsafe characters: {language}")

        effective_cwd = cwd or self.working_dir
        logger.debug(
            "language=<%s>, timeout=<%s>, cwd=<%s> | executing code locally",
            language,
            timeout,
            effective_cwd,
        )

        working_path = Path(effective_cwd)
        working_path.mkdir(parents=True, exist_ok=True)

        # Use create_subprocess_exec (not shell) — the interpreter and arguments
        # are passed directly to execvp, avoiding all shell quoting issues.
        try:
            proc = await asyncio.create_subprocess_exec(
                language,
                "-c",
                code,
                cwd=effective_cwd,
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
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Read stdout/stderr from a subprocess, then yield typed chunks and a final ExecutionResult.

        Shared helper used by both ``execute_streaming()`` and ``execute_code_streaming()`` to
        avoid duplicating the stream-reading, timeout-handling, and yielding logic.

        Args:
            proc: The running subprocess.
            timeout: Maximum time in seconds to wait for output. ``None`` means no timeout.

        Yields:
            :class:`StreamChunk` objects for stdout/stderr, then a final :class:`ExecutionResult`.

        Raises:
            asyncio.TimeoutError: If the process exceeds the timeout.
        """
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            read_task = asyncio.gather(
                _read_stream(proc.stdout, stdout_chunks),
                _read_stream(proc.stderr, stderr_chunks),
                proc.wait(),
            )
            await asyncio.wait_for(read_task, timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        for chunk in stdout_chunks:
            yield StreamChunk(data=chunk, stream_type="stdout")
        for chunk in stderr_chunks:
            yield StreamChunk(data=chunk, stream_type="stderr")

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
                    entries.append(FileInfo(name=name))
            return entries

        return await asyncio.to_thread(_list)
