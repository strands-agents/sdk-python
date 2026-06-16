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
- :class:`BubblewrapSandbox` — run commands inside an unprivileged ``bwrap``
  Linux user-namespace jail (no daemon, namespace-level isolation, Linux-only),
  configured via :class:`BubblewrapConfig` and :class:`BindMount`.
- Data types: :class:`StreamChunk`, :class:`FileInfo`, :class:`OutputFile`,
  :class:`ExecutionResult`, and the :data:`StreamType` literal.
- :data:`LANGUAGE_PATTERN` — interpreter-name validation pattern.

.. note::
   Per the "Prefer Flat Namespaces Over Nested Modules" decision record, the
   commonly-used symbols here will be re-exported from the top-level ``strands``
   package. That re-export is deferred to the Agent↔Sandbox integration
   follow-up (where the public surface stabilizes).

Example:
    A minimal shell-backed sandbox needs only ``execute_streaming``::

        from strands.sandbox import PosixShellSandbox

        class MyShellSandbox(PosixShellSandbox):
            async def execute_streaming(self, command, *, timeout=None, cwd=None, env=None, **kwargs):
                ...  # spawn a process, yield StreamChunk(s), then an ExecutionResult
"""

from .base import Sandbox
from .bubblewrap import (
    BindMount,
    BubblewrapConfig,
    BubblewrapSandbox,
    is_bubblewrap_available,
)
from .constants import LANGUAGE_PATTERN
from .docker import DockerSandbox
from .shell import PosixShellSandbox
from .ssh import SshSandbox
from .types import ExecutionResult, FileInfo, OutputFile, StreamChunk, StreamType

__all__ = [
    "BindMount",
    "BubblewrapConfig",
    "BubblewrapSandbox",
    "DockerSandbox",
    "ExecutionResult",
    "FileInfo",
    "LANGUAGE_PATTERN",
    "OutputFile",
    "PosixShellSandbox",
    "Sandbox",
    "SshSandbox",
    "StreamChunk",
    "StreamType",
    "is_bubblewrap_available",
]
