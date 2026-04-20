"""Sandbox abstraction for agent code execution environments.

This module provides the Sandbox interface that decouples tool logic from where code runs.
Tools that need to execute code or access a filesystem receive a Sandbox instead of managing
their own execution, enabling portability across local and cloud environments.

Class hierarchy::

    Sandbox (ABC, all abstract + lifecycle + helpers)
      ├── LocalSandbox — native Python methods for host execution (default)
      ├── ShellBasedSandbox (ABC, only execute_streaming() abstract — shell-based file ops + execute_code)
      └── NoOpSandbox — no-op implementation that disables all sandbox functionality
"""

from .base import ExecutionResult, FileInfo, OutputFile, Sandbox
from .local import LocalSandbox
from .noop import NoOpSandbox
from .shell_based import ShellBasedSandbox

__all__ = [
    "ExecutionResult",
    "FileInfo",
    "LocalSandbox",
    "NoOpSandbox",
    "OutputFile",
    "Sandbox",
    "ShellBasedSandbox",
]
