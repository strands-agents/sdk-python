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

from .base import ExecutionResult, Workspace

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

        Yields:
            str lines of output as they arrive, then a final ExecutionResult.
        """
        async for chunk in self.execute(
            f"{shlex.quote(language)} -c {shlex.quote(code)}", timeout=timeout
        ):
            yield chunk

    async def read_file(self, path: str) -> str:
        """Read a file from the workspace filesystem.

        Override for native file I/O support. The default implementation
        uses shell commands.

        Args:
            path: Path to the file to read.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        result = await self._execute_to_result(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the workspace filesystem.

        Override for native file I/O support. The default implementation
        uses a shell heredoc with a randomized delimiter to prevent
        content injection.

        Args:
            path: Path to the file to write.
            content: The content to write to the file.

        Raises:
            IOError: If the file cannot be written.
        """
        # Use a randomized heredoc delimiter to prevent injection when content
        # contains the delimiter string.
        delimiter = f"STRANDS_EOF_{secrets.token_hex(8)}"
        result = await self._execute_to_result(
            f"cat > {shlex.quote(path)} << '{delimiter}'\n{content}\n{delimiter}"
        )
        if result.exit_code != 0:
            raise IOError(result.stderr)

    async def remove_file(self, path: str) -> None:
        """Remove a file from the workspace filesystem.

        Override for native file removal support. The default implementation
        uses ``rm`` via the shell.

        Args:
            path: Path to the file to remove.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        result = await self._execute_to_result(f"rm {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)

    async def list_files(self, path: str = ".") -> list[str]:
        """List files in a workspace directory.

        Override for native directory listing support. The default
        implementation uses shell commands.

        Args:
            path: Path to the directory to list.

        Returns:
            A list of filenames in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        result = await self._execute_to_result(f"ls -1 {shlex.quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(result.stderr)
        return [f for f in result.stdout.strip().split("\n") if f]
