"""Base workspace interface for agent code execution environments.

This module defines the abstract Workspace class and the ExecutionResult dataclass.

Workspace implementations provide the runtime context where tools execute code, run commands,
and interact with a filesystem. Multiple tools share the same Workspace instance, giving them
a common working directory, environment variables, and filesystem.

Class hierarchy::

    Workspace (ABC): All 6 operations are abstract. Implement this for non-shell-based
        workspaces (e.g., API-based cloud workspaces).
    ShellBasedWorkspace (ABC, in shell_based.py): Provides shell-based defaults for file
        operations and code execution. Subclasses only need to implement ``execute()``.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code or command execution in a workspace.

    Attributes:
        exit_code: The exit code of the command or code execution.
        stdout: Standard output captured from execution.
        stderr: Standard error captured from execution.
    """

    exit_code: int
    stdout: str
    stderr: str


class Workspace(ABC):
    """Abstract execution environment for agent tools.

    A Workspace provides the runtime context where tools execute code,
    run commands, and interact with a filesystem. Multiple tools
    share the same Workspace instance, giving them a common working
    directory, environment variables, and filesystem.

    All six operations — ``execute``, ``execute_code``, ``read_file``,
    ``write_file``, ``remove_file``, and ``list_files`` — are abstract.
    Implement this directly for non-shell-based backends (e.g., API-driven
    cloud workspaces). For shell-based backends, extend
    :class:`~strands.workspace.shell_based.ShellBasedWorkspace` instead.

    The workspace auto-starts on the first ``execute()`` call if not already
    started, so callers do not need to manually call ``start()`` or use
    the async context manager.

    Example:
        ```python
        from strands.workspace import LocalWorkspace

        workspace = LocalWorkspace(working_dir="/tmp/my-workspace")
        async for chunk in workspace.execute("echo hello"):
            if isinstance(chunk, str):
                print(chunk, end="")  # stream output
        ```
    """

    def __init__(self) -> None:
        """Initialize base workspace state."""
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

        The workspace is auto-started on the first call if not already started.

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

    @abstractmethod
    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code in the workspace, streaming output.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds. None means no timeout.

        Yields:
            str lines of output as they arrive, then a final ExecutionResult.
        """
        ...
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file from the workspace filesystem.

        Args:
            path: Path to the file to read.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the workspace filesystem.

        Args:
            path: Path to the file to write.
            content: The content to write to the file.

        Raises:
            IOError: If the file cannot be written.
        """
        ...

    @abstractmethod
    async def remove_file(self, path: str) -> None:
        """Remove a file from the workspace filesystem.

        Args:
            path: Path to the file to remove.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        ...

    @abstractmethod
    async def list_files(self, path: str = ".") -> list[str]:
        """List files in a workspace directory.

        Args:
            path: Path to the directory to list.

        Returns:
            A list of filenames in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        ...

    async def _ensure_started(self) -> None:
        """Auto-start the workspace if it has not been started yet."""
        if not self._started:
            await self.start()
            self._started = True

    async def start(self) -> None:
        """Initialize the workspace.

        Called once before first use. Override to perform setup such as
        starting containers or creating temporary directories.
        """
        self._started = True

    async def stop(self) -> None:
        """Clean up workspace resources.

        Override to perform cleanup such as stopping containers or
        removing temporary directories.
        """
        self._started = False

    async def __aenter__(self) -> "Workspace":
        """Enter the async context manager, starting the workspace."""
        await self.start()
        self._started = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async context manager, stopping the workspace."""
        await self.stop()
        self._started = False
