"""Shell-based sandbox with default implementations for file and code operations.

This module defines the ShellBasedSandbox abstract class, which provides
shell-command-based defaults for file operations (read, write, remove, list)
and code execution. Subclasses only need to implement ``execute_streaming()``.

Use this for remote environments where only shell access is available
(e.g., Docker containers, SSH connections). For local execution, use
:class:`~strands.sandbox.host.HostSandbox` which uses native
Python methods instead.

Class hierarchy::

    Sandbox (ABC, all abstract)
      └── ShellBasedSandbox (ABC, only execute_streaming() abstract — shell-based file ops + execute_code)
"""

import base64
import logging
import shlex
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any

from .base import ExecutionResult, FileInfo, Sandbox, StreamChunk

logger = logging.getLogger(__name__)


class ShellBasedSandbox(Sandbox, ABC):
    """Abstract sandbox that provides shell-based defaults for file and code operations.

    Subclasses only need to implement :meth:`execute_streaming`. The remaining
    operations — ``execute_code_streaming``, ``read_file``, ``write_file``,
    ``remove_file``, and ``list_files`` — are implemented via shell commands
    piped through ``execute_streaming()``.

    This class is intended for remote execution environments where only
    shell access is available (e.g., Docker containers, SSH connections).
    For local execution, use :class:`~strands.sandbox.host.HostSandbox`
    which uses native Python methods for better safety and reliability.

    Subclasses may override any method with a native implementation for
    better performance.
    """

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute code in the sandbox, streaming output.

        The default implementation passes code to the language interpreter
        via ``-c`` with proper shell quoting. Both the ``language`` and
        ``code`` parameters are sanitized with :func:`shlex.quote` to
        prevent command injection.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use (e.g.
                ``"python"``, ``"node"``, ``"ruby"``).
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for code execution. ``None`` means use the
                sandbox's default working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final :class:`ExecutionResult`.

        Note:
            The default implementation assumes the language interpreter
            accepts code via the ``-c`` flag (e.g., ``python -c "code"``).
            Override this method for interpreters that require a different
            invocation pattern (e.g., ``javac``, ``gcc``, ``go run``).
        """
        async for chunk in self.execute_streaming(
            f"{shlex.quote(language)} -c {shlex.quote(code)}", timeout=timeout, cwd=cwd
        ):
            yield chunk

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the sandbox filesystem as raw bytes.

        Uses ``base64`` to encode the file content for safe transport
        through the shell text layer, then decodes on the Python side.
        This preserves binary content (images, PDFs, compiled files)
        that would be corrupted by direct ``cat`` through a text pipe.

        Override for native file I/O support if the backend provides
        a binary-safe channel (e.g., Docker stdin/stdout pipes).

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        result = await self.execute(f"base64 {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)
        # base64 output is ASCII-safe text — decode it back to raw bytes
        return base64.b64decode(result.stdout)

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write bytes to a file in the sandbox filesystem.

        Uses ``base64`` encoding to safely transport binary content through
        the shell text layer. The encoded data is piped through ``base64 -d``
        to decode it back to raw bytes on the remote side. Parent directories
        are created automatically via ``mkdir -p``.

        This approach handles any content type (text, images, PDFs, compiled
        files) without corruption. Override for native file I/O if the
        backend provides a binary-safe channel.

        Note:
            The base64-encoded content is passed as a shell argument to
            ``printf``. For very large files (roughly >1.5 MB of original
            content, which becomes ~2 MB after base64 encoding), this may
            exceed the shell's ``ARG_MAX`` limit on some systems. For large
            binary files, override this method with a stdin-piping approach
            or use a binary-safe channel.

        Args:
            path: Path to the file to write.
            content: The content to write as bytes.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            IOError: If the file cannot be written.
        """
        encoded = base64.b64encode(content).decode("ascii")
        quoted_path = shlex.quote(path)
        # Create parent directories, then write content via base64 decode
        cmd = f"mkdir -p \"$(dirname {quoted_path})\" && printf '%s' {shlex.quote(encoded)} | base64 -d > {quoted_path}"
        result = await self.execute(cmd)
        if result.exit_code != 0:
            raise OSError(result.stderr)

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the sandbox filesystem.

        Override for native file removal support. The default implementation
        uses ``rm`` via the shell.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        result = await self.execute(f"rm {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a sandbox directory with structured metadata.

        Uses ``ls -1aF`` to include hidden files (dotfiles) and identify
        directories. Returns :class:`FileInfo` entries with name and is_dir.
        Size is ``None`` for shell-based listing (cannot be determined
        reliably from ``ls -1aF`` output alone).

        Override for native directory listing support.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        result = await self.execute(f"ls -1aF {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)

        entries = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or line in (".", "..", "./", "../"):
                continue
            # ls -F appends / for directories, @ for symlinks, * for executables
            is_dir = line.endswith("/")
            # Strip the type indicator from the name
            name = line.rstrip("/@*=|")
            if name:
                entries.append(FileInfo(name=name, is_dir=is_dir))
        return entries
