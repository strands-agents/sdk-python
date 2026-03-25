"""Base sandbox interface for agent code execution environments.

This module defines the abstract Sandbox class and the ExecutionResult dataclass.
Sandbox implementations provide the runtime context where tools execute code, run commands,
and interact with a filesystem. Multiple tools share the same Sandbox instance, giving them
a common working directory, environment variables, and filesystem.

Implementations only need to provide execute(). All other methods are built on top of it.
Implementations may override convenience methods with native versions for better performance.
"""

import logging
import secrets
import shlex
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code or command execution in a sandbox.

    Attributes:
        exit_code: The exit code of the command or code execution.
        stdout: Standard output captured from execution.
        stderr: Standard error captured from execution.
    """

    exit_code: int
    stdout: str
    stderr: str


class Sandbox(ABC):
    """Abstract execution environment for agent tools.

    A Sandbox provides the runtime context where tools execute code,
    run commands, and interact with a filesystem. Multiple tools
    share the same Sandbox instance, giving them a common working
    directory, environment variables, and filesystem.

    Implementations only need to provide execute(). All other methods
    are built on top of it. Implementations may override convenience
    methods with native versions for better performance (for example,
    LocalSandbox overrides read_file/write_file with native file I/O).

    The sandbox auto-starts on the first ``execute()`` call if not already
    started, so callers do not need to manually call ``start()`` or use
    the async context manager.

    Example:
        ```python
        from strands.sandbox import LocalSandbox

        sandbox = LocalSandbox(working_dir="/tmp/workspace")
        async for chunk in sandbox.execute("echo hello"):
            if isinstance(chunk, str):
                print(chunk, end="")  # stream output
        ```
    """

    def __init__(self) -> None:
        """Initialize base sandbox state."""
        self._started = False

    @abstractmethod
    async def execute(
        self,
        command: str,
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute a shell command, streaming output.

        Yields stdout/stderr lines as they arrive. The final yield
        is an ExecutionResult with the exit code and complete output.

        This is the only method implementations must provide. All other
        methods are built on top of this one by default.

        The sandbox is auto-started on the first call if not already started.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. None means no timeout.

        Yields:
            str lines of output as they arrive, then a final ExecutionResult.
        """
        ...
        # Make the method signature an async generator for type checkers.
        # Concrete subclasses must yield at least one ExecutionResult.
        yield  # type: ignore[misc]  # pragma: no cover

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code in the sandbox, streaming output.

        Override for native code execution support. The default implementation
        passes code to the language interpreter via ``-c`` with proper shell
        quoting.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds. None means no timeout.

        Yields:
            str lines of output as they arrive, then a final ExecutionResult.
        """
        async for chunk in self.execute(f"{language} -c {shlex.quote(code)}", timeout=timeout):
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

    async def _ensure_started(self) -> None:
        """Auto-start the sandbox if it has not been started yet."""
        if not self._started:
            await self.start()
            self._started = True

    async def start(self) -> None:
        """Initialize the sandbox.

        Called once before first use. Override to perform setup such as
        starting containers or creating temporary directories.
        """
        self._started = True

    async def stop(self) -> None:
        """Clean up sandbox resources.

        Override to perform cleanup such as stopping containers or
        removing temporary directories.
        """
        self._started = False

    async def __aenter__(self) -> "Sandbox":
        """Enter the async context manager, starting the sandbox."""
        await self.start()
        self._started = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager, stopping the sandbox."""
        await self.stop()
        self._started = False
