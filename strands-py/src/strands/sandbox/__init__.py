"""Sandbox abstraction for agent code-execution environments.

A :class:`Sandbox` provides the runtime context where tools execute code, run
commands, and interact with a filesystem. This module ports the sandbox
interface from ``strands-ts/src/sandbox/`` (the behavioral oracle):

- :class:`Sandbox` — the abstract base with streaming primitives and
  non-streaming/text convenience wrappers.
- :class:`PosixShellSandbox` — an abstract sandbox that implements file and code
  operations via shell commands; subclasses implement only
  :meth:`~strands.sandbox.base.Sandbox.execute_streaming`.
- :class:`DockerSandbox` — run commands in a Docker container via ``docker exec``.
- :class:`SshSandbox` — run commands on a remote host via OpenSSH.
- Data types: :class:`StreamChunk`, :class:`FileInfo`, :class:`OutputFile`,
  :class:`ExecutionResult`, and the :data:`StreamType` literal.
- :data:`LANGUAGE_PATTERN` — interpreter-name validation pattern.

Commonly-used symbols are also re-exported from the top-level ``strands`` package.

Example:
    A minimal shell-backed sandbox needs only ``execute_streaming``::

        from strands.sandbox import PosixShellSandbox

        class MyShellSandbox(PosixShellSandbox):
            async def execute_streaming(self, command, *, timeout=None, cwd=None, env=None, **kwargs):
                ...  # spawn a process, yield StreamChunk(s), then an ExecutionResult
"""

from .base import Sandbox
from .constants import LANGUAGE_PATTERN
from .docker import DockerSandbox
from .not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from .posix_shell import PosixShellSandbox
from .ssh import SshSandbox
from .types import ExecutionResult, FileInfo, OutputFile, StreamChunk, StreamType

# One shared, stateless instance, mirroring the TS oracle's module-level default.
_default_sandbox = NotASandboxLocalEnvironment()


def default_sandbox() -> Sandbox:
    """Return the shared host default used when an Agent has no sandbox configured."""
    return _default_sandbox


__all__ = [
    "DockerSandbox",
    "ExecutionResult",
    "FileInfo",
    "LANGUAGE_PATTERN",
    "NotASandboxLocalEnvironment",
    "OutputFile",
    "PosixShellSandbox",
    "Sandbox",
    "SshSandbox",
    "StreamChunk",
    "StreamType",
]
