"""No-op sandbox implementation that disables all sandbox functionality.

Use ``NoOpSandbox`` to explicitly disable sandbox features on an agent.
All operations raise ``NotImplementedError`` with a clear message.

Example::

    from strands import Agent
    from strands.sandbox import NoOpSandbox

    # Explicitly disable sandbox functionality
    agent = Agent(sandbox=NoOpSandbox())
"""

from collections.abc import AsyncGenerator
from typing import Any

from .base import ExecutionResult, FileInfo, Sandbox, StreamChunk


class NoOpSandbox(Sandbox):
    """No-op sandbox that raises NotImplementedError for all operations.

    Use this to explicitly disable sandbox functionality on an agent.
    Any tool that attempts to use the sandbox will get a clear error
    indicating that sandbox is disabled, rather than silently failing.

    Example::

        from strands import Agent
        from strands.sandbox import NoOpSandbox

        agent = Agent(sandbox=NoOpSandbox())
    """

    async def execute_streaming(
        self,
        command: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot execute commands.")
        yield  # type: ignore[unreachable]  # pragma: no cover

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot execute code.")
        yield  # type: ignore[unreachable]  # pragma: no cover

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot read files.")

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot write files.")

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot remove files.")

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """Raise NotImplementedError — sandbox is disabled."""
        raise NotImplementedError("Sandbox is disabled (NoOpSandbox). Cannot list files.")
