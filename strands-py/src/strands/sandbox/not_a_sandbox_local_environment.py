"""Host execution environment used as the default when no sandbox is configured.

:class:`NotASandboxLocalEnvironment` runs commands, code, and file operations
directly on the host with **no isolation**. The deliberately blunt name (mirrored
from ``strands-ts/src/sandbox/not-a-sandbox-local-environment.ts``) is a warning:
this is the fallback an :class:`~strands.agent.agent.Agent` uses when no sandbox
is passed, not a security boundary.

Idiomatic divergences from the TypeScript oracle:

- TS's ``NotASandboxLocalEnvironment`` extends ``Sandbox`` directly and reimplements
  the base64 heredoc code-execution logic. Python already factored that into
  :class:`~strands.sandbox.posix_shell.PosixShellSandbox`, so this class extends it,
  inherits :meth:`~strands.sandbox.posix_shell.PosixShellSandbox.execute_code_streaming`,
  and implements only the shell :meth:`execute_streaming`.
- File operations are overridden with **native** :mod:`pathlib`/:mod:`os` calls
  rather than the shell-based defaults, so they avoid spawning a shell and report
  real ``size`` metadata in :meth:`list_files` (the shell-based base always returns
  ``None``).
"""

import os
import shlex
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from .posix_shell import PosixShellSandbox, build_shell_env_prefix
from .stream_process import stream_process
from .types import ExecutionResult, FileInfo, StreamChunk


class NotASandboxLocalEnvironment(PosixShellSandbox):
    """Run commands, code, and file operations on the host with no isolation.

    Used as the default execution environment when an :class:`Agent` is created
    without a ``sandbox``. Command and code execution spawn a local ``sh``; file
    operations use the host filesystem directly.

    .. warning::
        This provides **no isolation**. Commands run with the full privileges of
        the host process. Pass an explicit sandbox (e.g.
        :class:`~strands.sandbox.docker.DockerSandbox`) when isolation matters.
    """

    @staticmethod
    def _resolve_path(path: str) -> Path:
        """Resolve ``path`` against the current working directory if relative."""
        return Path(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a command on the host via ``sh -c``, streaming output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for this command. Defaults to the process's
                current working directory.
            env: Environment variables to set, applied via a shell ``export`` prefix.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final
            :class:`ExecutionResult`.

        Raises:
            ValueError: If an environment variable name is invalid.
            TimeoutError: If execution exceeds ``timeout`` seconds.
        """
        target_cwd = cwd if cwd is not None else os.getcwd()
        env_prefix = build_shell_env_prefix(env)
        full_command = f"cd {shlex.quote(target_cwd)} && {env_prefix}{command}"
        async for chunk in stream_process("sh", ["-c", full_command], timeout=timeout):
            yield chunk

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the host filesystem as raw bytes.

        Args:
            path: Path to the file. Relative paths resolve against the current
                working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as raw bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError: If the file cannot be read.
        """
        return self._resolve_path(path).read_bytes()

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write raw bytes to a file on the host, creating parent directories.

        Args:
            path: Path to the file. Relative paths resolve against the current
                working directory.
            content: The content to write.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            OSError: If the file cannot be written.
        """
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the host filesystem.

        Args:
            path: Path to the file. Relative paths resolve against the current
                working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._resolve_path(path).unlink()

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List directory contents from the host filesystem, sorted by name.

        Unlike the shell-based base implementation, this reports native ``is_dir``
        and ``size`` metadata. If an entry's metadata cannot be read, it is still
        listed with ``is_dir``/``size`` left as ``None``.

        Args:
            path: Path to the directory. Relative paths resolve against the
                current working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries for the directory contents.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If ``path`` is not a directory.
        """
        full_path = self._resolve_path(path)
        results: list[FileInfo] = []
        with os.scandir(full_path) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                try:
                    stat = entry.stat()
                    results.append(FileInfo(name=entry.name, is_dir=entry.is_dir(), size=stat.st_size))
                except OSError:
                    results.append(FileInfo(name=entry.name))
        return results
