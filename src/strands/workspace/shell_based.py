"""Shell-based workspace with default implementations for file and code operations.

This module defines the ShellBasedWorkspace abstract class, which provides
shell-command-based defaults for file operations (read, write, remove, list)
and code execution. Subclasses only need to implement ``execute()``.

Use this for remote environments where only shell access is available
(e.g., Docker containers, SSH connections). For local execution, use
:class:`~strands.workspace.local.LocalWorkspace` which uses native
Python methods instead.

Class hierarchy::

    Workspace (ABC, all 6 abstract + lifecycle)
      └── ShellBasedWorkspace (ABC, only execute() abstract — shell-based file ops + execute_code)
"""

import logging
import secrets
import shlex
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any

from .base import ExecutionResult, FileInfo, Workspace

logger = logging.getLogger(__name__)


class ShellBasedWorkspace(Workspace, ABC):
    """Abstract workspace that provides shell-based defaults for file and code operations.

    Subclasses only need to implement :meth:`execute`. The remaining five
    operations — ``read_file``, ``write_file``, ``remove_file``,
    ``list_files``, and ``execute_code`` — are implemented via shell
    commands piped through ``execute()``.

    This class is intended for remote execution environments where only
    shell access is available (e.g., Docker containers, SSH connections).
    For local execution, use :class:`~strands.workspace.local.LocalWorkspace`
    which uses native Python methods for better safety and reliability.

    Subclasses may override any method with a native implementation for
    better performance.
    """

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code in the workspace, streaming output.

        The default implementation passes code to the language interpreter
        via ``-c`` with proper shell quoting. Both the ``language`` and
        ``code`` parameters are sanitized with :func:`shlex.quote` to
        prevent command injection.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use (e.g.
                ``"python"``, ``"node"``, ``"ruby"``).
            timeout: Maximum execution time in seconds. None means no timeout.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            str chunks of output, then a final ExecutionResult.
        """
        async for chunk in self.execute(
            f"{shlex.quote(language)} -c {shlex.quote(code)}", timeout=timeout
        ):
            yield chunk

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the workspace filesystem as raw bytes.

        Override for native file I/O support. The default implementation
        uses shell commands. Uses ``cat`` to read the file, which outputs
        raw bytes.

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        result = await self._execute_to_result(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)
        # Shell stdout is captured as str; encode back to bytes
        return result.stdout.encode("utf-8")

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write bytes to a file in the workspace filesystem.

        Override for native file I/O support. The default implementation
        uses a shell heredoc with a randomized delimiter to prevent
        content injection. Note: this shell-based approach works best
        with text content. For truly binary content, subclasses should
        override with a native implementation.

        Args:
            path: Path to the file to write.
            content: The content to write as bytes.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            IOError: If the file cannot be written.
        """
        # Decode bytes to text for heredoc-based writing.
        # For binary content, subclasses should override with native I/O.
        text_content = content.decode("utf-8")
        # Use a randomized heredoc delimiter to prevent injection when content
        # contains the delimiter string.
        delimiter = f"STRANDS_EOF_{secrets.token_hex(8)}"
        result = await self._execute_to_result(
            f"cat > {shlex.quote(path)} << '{delimiter}'\n{text_content}\n{delimiter}"
        )
        if result.exit_code != 0:
            raise IOError(result.stderr)

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the workspace filesystem.

        Override for native file removal support. The default implementation
        uses ``rm`` via the shell.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        result = await self._execute_to_result(f"rm {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a workspace directory with structured metadata.

        Uses ``ls -1aF`` to include hidden files (dotfiles) and identify
        directories. Returns :class:`FileInfo` entries with name, is_dir,
        and size (size defaults to 0 for shell-based listing).

        Override for native directory listing support.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        result = await self._execute_to_result(f"ls -1aF {shlex.quote(path)}")
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
                entries.append(FileInfo(name=name, is_dir=is_dir, size=0))
        return entries
