"""Base workspace interface for agent code execution environments.

This module defines the abstract Workspace class and supporting dataclasses:

- :class:`ExecutionResult` — result of command/code execution
- :class:`FileInfo` — metadata about a file in the workspace
- :class:`OutputFile` — a file produced as output by code execution

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
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Metadata about a file or directory in a workspace.

    Provides minimal structured information that lets tools distinguish
    files from directories and report sizes. Matches the pattern used
    by OpenAI (FileEntry), E2B (EntryInfo), and Daytona (FileInfo).

    Attributes:
        name: The file or directory name (not the full path).
        is_dir: Whether this entry is a directory.
        size: File size in bytes. Defaults to 0 for directories
            or when size is unknown.
    """

    name: str
    is_dir: bool
    size: int = 0


@dataclass
class OutputFile:
    """A file produced as output by code execution.

    Used to carry binary artifacts (images, charts, PDFs, compiled files)
    from workspace execution back to the agent. Tools can convert these
    to Strands' ``ImageContent`` or ``DocumentContent`` for the model.

    Follows ADK's ``File`` pattern — simple, portable, MIME-typed.

    Attributes:
        name: Filename (e.g., ``"plot.png"``).
        content: Raw file content as bytes.
        mime_type: MIME type of the content (e.g., ``"image/png"``).
    """

    name: str
    content: bytes
    mime_type: str = "application/octet-stream"


@dataclass
class ExecutionResult:
    """Result of code or command execution in a workspace.

    Attributes:
        exit_code: The exit code of the command or code execution.
        stdout: Standard output captured from execution.
        stderr: Standard error captured from execution.
        output_files: Files produced by the execution (e.g., images, charts).
            Shell-based workspaces typically return an empty list. Jupyter-backed
            or API-backed workspaces can populate this with generated artifacts.
    """

    exit_code: int
    stdout: str
    stderr: str
    output_files: list[OutputFile] = field(default_factory=list)


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

    All abstract methods accept ``**kwargs`` for forward compatibility —
    new parameters with defaults can be added in future versions without
    breaking existing implementations.

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
        **kwargs: Any,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute a shell command, streaming output.

        Yields collected stdout/stderr chunks after the command completes.
        The final yield is an ExecutionResult with the exit code and
        complete output.

        The workspace is auto-started on the first call if not already started.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. None means no timeout.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            str chunks of output, then a final ExecutionResult.
        """
        ...
        # Make the method signature an async generator for type checkers.
        # Concrete subclasses must yield at least one ExecutionResult.
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str | ExecutionResult, None]:
        """Execute code in the workspace, streaming output.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds. None means no timeout.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            str chunks of output, then a final ExecutionResult.
        """
        ...
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the workspace filesystem.

        Returns raw bytes to support both text and binary files (images,
        PDFs, compiled artifacts). Use :meth:`read_text` for a convenience
        wrapper that decodes to a string.

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        ...

    @abstractmethod
    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write a file to the workspace filesystem.

        Accepts raw bytes to support both text and binary content. Use
        :meth:`write_text` for a convenience wrapper that encodes a string.

        Args:
            path: Path to the file to write.
            content: The content to write as bytes.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            IOError: If the file cannot be written.
        """
        ...

    @abstractmethod
    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the workspace filesystem.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        ...

    @abstractmethod
    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a workspace directory.

        Returns structured :class:`FileInfo` entries with metadata (name,
        is_dir, size) so tools can make informed decisions about files.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries for the directory contents.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        ...

    # ---- Convenience methods (non-abstract) ----

    async def read_text(self, path: str, encoding: str = "utf-8", **kwargs: Any) -> str:
        """Read a text file from the workspace filesystem.

        Convenience wrapper around :meth:`read_file` that decodes bytes
        to a string.

        Args:
            path: Path to the file to read.
            encoding: Text encoding to use. Defaults to UTF-8.
            **kwargs: Additional keyword arguments passed to :meth:`read_file`.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file cannot be decoded with the given encoding.
        """
        data = await self.read_file(path, **kwargs)
        return data.decode(encoding)

    async def write_text(self, path: str, content: str, encoding: str = "utf-8", **kwargs: Any) -> None:
        """Write a text file to the workspace filesystem.

        Convenience wrapper around :meth:`write_file` that encodes a string
        to bytes.

        Args:
            path: Path to the file to write.
            content: The text content to write.
            encoding: Text encoding to use. Defaults to UTF-8.
            **kwargs: Additional keyword arguments passed to :meth:`write_file`.

        Raises:
            IOError: If the file cannot be written.
        """
        await self.write_file(path, content.encode(encoding), **kwargs)

    async def _execute_to_result(self, command: str, timeout: int | None = None) -> ExecutionResult:
        """Helper: consume the execute() stream and return the final ExecutionResult.

        Convenience methods use this to get just the final result without
        dealing with the stream.

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
