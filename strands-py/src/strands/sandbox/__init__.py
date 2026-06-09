"""Sandbox abstraction for agent code-execution environments.

A :class:`Sandbox` provides the runtime context where tools execute code, run
commands, and interact with a filesystem. This module ports the *core* sandbox
interface from ``strands-ts/src/sandbox/`` (the behavioral oracle):

- :class:`Sandbox` — the abstract base with streaming primitives and
  non-streaming/text convenience wrappers.
- :class:`PosixShellSandbox` — an abstract sandbox that implements file and code
  operations via shell commands; subclasses implement only
  :meth:`~strands.sandbox.base.Sandbox.execute_streaming`.
- Data types: :class:`StreamChunk`, :class:`FileInfo`, :class:`OutputFile`,
  :class:`ExecutionResult`, and the :data:`StreamType` literal.
- :data:`LANGUAGE_PATTERN` — interpreter-name validation pattern.

Concrete shell backends (Docker, SSH) and the Agent↔Sandbox integration are
intentionally out of scope for this "core only (1/N)" port; they follow as
separate PRs mirroring the corresponding TypeScript modules.

.. note::
   Per the "Prefer Flat Namespaces Over Nested Modules" decision record, the
   commonly-used symbols here will be re-exported from the top-level ``strands``
   package. That re-export is deferred to the Agent↔Sandbox integration
   follow-up (where the public surface stabilizes), so it is not added in this
   core-only PR.

Example:
    A minimal shell-backed sandbox needs only ``execute_streaming``::

        from strands.sandbox import PosixShellSandbox

        class MyShellSandbox(PosixShellSandbox):
            async def execute_streaming(self, command, *, timeout=None, cwd=None, env=None, **kwargs):
                ...  # spawn a process, yield StreamChunk(s), then an ExecutionResult
"""

from .base import Sandbox
from .constants import LANGUAGE_PATTERN
from .shell import PosixShellSandbox
from .types import ExecutionResult, FileInfo, OutputFile, StreamChunk, StreamType

__all__ = [
    "ExecutionResult",
    "FileInfo",
    "LANGUAGE_PATTERN",
    "OutputFile",
    "PosixShellSandbox",
    "Sandbox",
    "StreamChunk",
    "StreamType",
]
