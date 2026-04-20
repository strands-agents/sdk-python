"""Base sandbox interface for agent code execution environments.

This module defines the abstract Sandbox class and supporting dataclasses:

- :class:`ExecutionResult` — result of command/code execution
- :class:`FileInfo` — metadata about a file in the sandbox
- :class:`OutputFile` — a file produced as output by code execution
- :class:`StreamChunk` — a typed chunk of streaming output (stdout or stderr)

Sandbox implementations provide the runtime context where tools execute code, run commands,
and interact with a filesystem. Multiple tools share the same Sandbox instance, giving them
a common working directory, environment variables, and filesystem.

Class hierarchy::

    Sandbox (ABC): All operations are abstract. Implement this for non-shell-based
        sandboxes (e.g., API-based cloud sandboxes).
    ShellBasedSandbox (ABC, in shell_based.py): Provides shell-based defaults for file
        operations and code execution. Subclasses only need to implement ``execute_streaming()``.
    NoOpSandbox (in noop.py): No-op implementation that raises NotImplementedError
        for all operations. Use to disable sandbox functionality entirely.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Type of a streaming output chunk.

    Used by :class:`StreamChunk` to distinguish stdout from stderr output
    during streaming execution.
    """

    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass
class StreamChunk:
    """A typed chunk of streaming output from command or code execution.

    Allows consumers to distinguish stdout from stderr during streaming,
    enabling richer UIs and more precise output handling.

    Attributes:
        data: The text content of the chunk.
        stream_type: Whether this chunk is from stdout or stderr.
    """

    data: str
    stream_type: StreamType = StreamType.STDOUT


@dataclass
class FileInfo:
    """Metadata about a file or directory in a sandbox.

    Provides minimal structured information that lets tools distinguish
    files from directories and report sizes. Fields ``is_dir`` and ``size``
    are optional — implementations that cannot provide accurate data
    return ``None`` instead of lying.

    Attributes:
        name: The file or directory name (not the full path).
        is_dir: Whether this entry is a directory. ``None`` if unknown.
        size: File size in bytes. ``None`` if unknown.
    """

    name: str
    is_dir: bool | None = None
    size: int | None = None


@dataclass
class OutputFile:
    """A file produced as output by code execution.

    Used to carry binary artifacts (images, charts, PDFs, compiled files)
    from sandbox execution back to the agent. Tools can convert these
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
    """Result of code or command execution in a sandbox.

    Attributes:
        exit_code: The exit code of the command or code execution.
        stdout: Standard output captured from execution.
        stderr: Standard error captured from execution.
        output_files: Files produced by the execution (e.g., images, charts).
            Shell-based sandboxes typically return an empty list. Jupyter-backed
            or API-backed sandboxes can populate this with generated artifacts.
    """

    exit_code: int
    stdout: str
    stderr: str
    output_files: list[OutputFile] = field(default_factory=list)


class Sandbox(ABC):
    """Abstract execution environment for agent tools.

    A Sandbox provides the runtime context where tools execute code,
    run commands, and interact with a filesystem. Multiple tools
    share the same Sandbox instance, giving them a common working
    directory, environment variables, and filesystem.

    The sandbox follows the SDK's ``invoke_async`` / ``stream_async``
    pattern: streaming methods (``execute_streaming``, ``execute_code_streaming``)
    are the abstract primitives that implementations must provide.
    Non-streaming convenience methods (``execute``, ``execute_code``) consume
    the stream and return the final ``ExecutionResult``.

    Streaming methods yield :class:`StreamChunk` objects that carry both
    the text data and the stream type (stdout or stderr), followed by a
    final :class:`ExecutionResult`. This allows consumers to distinguish
    between stdout and stderr during streaming.

    All abstract methods accept ``**kwargs`` for forward compatibility —
    new parameters with defaults can be added in future versions without
    breaking existing implementations.

    Example:
        Non-streaming (common case)::

            from strands.sandbox import LocalSandbox

            sandbox = LocalSandbox(working_dir="/tmp/my-sandbox")
            result = await sandbox.execute("echo hello")
            print(result.stdout)

        Streaming with stdout/stderr distinction::

            async for chunk in sandbox.execute_streaming("echo hello"):
                if isinstance(chunk, StreamChunk):
                    if chunk.stream_type == StreamType.STDOUT:
                        print(f"[stdout] {chunk.data}", end="")
                    else:
                        print(f"[stderr] {chunk.data}", end="")
                elif isinstance(chunk, ExecutionResult):
                    print(f"Exit code: {chunk.exit_code}")

        With custom working directory::

            result = await sandbox.execute("ls -la", cwd="/tmp/other-dir")
    """

    # ---- Streaming methods (abstract primitives) ----

    @abstractmethod
    async def execute_streaming(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a shell command, streaming output.

        Yields :class:`StreamChunk` objects for stdout and stderr output
        as it arrives. The final yield is an :class:`ExecutionResult` with
        the exit code and complete output.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for command execution. ``None`` means use the
                sandbox's default working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final :class:`ExecutionResult`.
        """
        ...
        # Make the method signature an async generator for type checkers.
        # Concrete subclasses must yield at least one ExecutionResult.
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute code in the sandbox, streaming output.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for code execution. ``None`` means use the
                sandbox's default working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final :class:`ExecutionResult`.
        """
        ...
        yield  # type: ignore[misc]  # pragma: no cover

    @abstractmethod
    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the sandbox filesystem.

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
        """Write a file to the sandbox filesystem.

        Accepts raw bytes to support both text and binary content. Use
        :meth:`write_text` for a convenience wrapper that encodes a string.

        Implementations should create parent directories if they do not exist.
        :class:`~strands.sandbox.local.LocalSandbox` does this natively
        via :func:`pathlib.Path.mkdir`. Shell-based implementations should
        include a ``mkdir -p`` before writing.

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
        """Remove a file from the sandbox filesystem.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        ...

    @abstractmethod
    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a sandbox directory.

        Returns structured :class:`FileInfo` entries with metadata (name,
        is_dir, size) so tools can make informed decisions about files.
        Fields ``is_dir`` and ``size`` may be ``None`` if the implementation
        cannot provide accurate data.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`FileInfo` entries for the directory contents.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        ...

    # ---- Non-streaming convenience methods ----

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a shell command and return the result.

        Convenience wrapper that consumes :meth:`execute_streaming` and
        returns the final :class:`ExecutionResult`. This is the common case —
        use :meth:`execute_streaming` when you need to process output as it
        arrives.

        Implementations that want an optimized non-streaming path can
        override this method directly.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for command execution. ``None`` means use the
                sandbox's default working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The final ExecutionResult from execution.

        Raises:
            RuntimeError: If execute_streaming() did not yield an ExecutionResult.
        """
        result = None
        async for chunk in self.execute_streaming(command, timeout=timeout, cwd=cwd, **kwargs):
            if isinstance(chunk, ExecutionResult):
                result = chunk
        if result is None:
            raise RuntimeError("execute_streaming() did not yield an ExecutionResult")
        return result

    async def execute_code(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute code and return the result.

        Convenience wrapper that consumes :meth:`execute_code_streaming` and
        returns the final :class:`ExecutionResult`. This is the common case —
        use :meth:`execute_code_streaming` when you need to process output as
        it arrives.

        Implementations that want an optimized non-streaming path can
        override this method directly.

        Args:
            code: The source code to execute.
            language: The programming language interpreter to use.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for code execution. ``None`` means use the
                sandbox's default working directory.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The final ExecutionResult from execution.

        Raises:
            RuntimeError: If execute_code_streaming() did not yield an ExecutionResult.
        """
        result = None
        async for chunk in self.execute_code_streaming(code, language=language, timeout=timeout, cwd=cwd, **kwargs):
            if isinstance(chunk, ExecutionResult):
                result = chunk
        if result is None:
            raise RuntimeError("execute_code_streaming() did not yield an ExecutionResult")
        return result

    # ---- Text convenience methods ----

    async def read_text(self, path: str, encoding: str = "utf-8", **kwargs: Any) -> str:
        """Read a text file from the sandbox filesystem.

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
        """Write a text file to the sandbox filesystem.

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
