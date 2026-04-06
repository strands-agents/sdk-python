"""Shell-based sandbox with default implementations for file and code operations.

This module defines the ShellBasedSandbox abstract class, which provides
shell-command-based defaults for file operations (read, write, remove, list)
and code execution. Subclasses only need to implement ``execute()``.

Class hierarchy::

    Sandbox (ABC, all 6 abstract + lifecycle)
      └── ShellBasedSandbox (ABC, only execute() abstract — shell-based file ops + execute_code)
            └── LocalSandbox
"""

import logging
import secrets
import shlex
from abc import ABC
from collections.abc import AsyncGenerator

from .base import ExecutionResult, Sandbox

logger = logging.getLogger(__name__)


class ShellBasedSandbox(Sandbox, ABC):
    """Abstract sandbox that provides shell-based defaults for file and code operations.

    Subclasses only need to implement :meth:`execute`. The remaining five
    operations — ``read_file``, ``write_file``, ``remove_file``,
    ``list_files``, and ``execute_code`` — are implemented via shell
    commands piped through ``execute()``.

    Subclasses may override any method with a native implementation for
    better performance (e.g., ``LocalSandbox`` overrides ``read_file``,
    ``write_file``, and ``remove_file`` with direct filesystem calls).
    """

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code in the sandbox, streaming output.

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

    async def _execute_to_result(self, command: str, timeout: int | None = None) -> ExecutionResult:
        """Helper: consume the execute() stream and return the final ExecutionResult.

        Convenience methods like read_file, write_file, and list_files use
        this to get just the final result without dealing with the stream.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            The final ExecutionResult from the stream.

        Raises:
            RuntimeError: If execute() did not yield an ExecutionResult.
        """
        result = None
        async for chunk in self.execute(command, timeout=timeout):
            if isinstance(chunk, ExecutionResult):
                result = chunk
        if result is None:
            raise RuntimeError("execute() did not yield an ExecutionResult")
        return result

    async def _execute_code_to_result(
        self, code: str, language: str = "python", timeout: int | None = None
    ) -> ExecutionResult:
        """Helper: consume the execute_code() stream and return the final ExecutionResult.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds.

        Returns:
            The final ExecutionResult from the stream.

        Raises:
            RuntimeError: If execute_code() did not yield an ExecutionResult.
        """
        result = None
        async for chunk in self.execute_code(code, language=language, timeout=timeout):
            if isinstance(chunk, ExecutionResult):
                result = chunk
        if result is None:
            raise RuntimeError("execute_code() did not yield an ExecutionResult")
        return result

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox filesystem.

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
        """Write a file to the sandbox filesystem.

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
        """Remove a file from the sandbox filesystem.

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
        """List files in a sandbox directory.

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
