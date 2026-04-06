"""Sandbox abstraction for agent code execution environments.

This module provides the Sandbox interface that decouples tool logic from where code runs.
Tools that need to execute code or access a filesystem receive a Sandbox instead of managing
their own execution, enabling portability across local and cloud environments.

Class hierarchy::

    Sandbox (ABC, all 6 abstract + lifecycle)
      └── ShellBasedSandbox (ABC, only execute() abstract — shell-based file ops + execute_code)
            └── LocalSandbox — runs on the host via asyncio subprocesses (default)
"""

from .base import ExecutionResult, Sandbox
from .local import LocalSandbox
from .shell_based import ShellBasedSandbox

__all__ = [
    "ExecutionResult",
    "LocalSandbox",
    "Sandbox",
    "ShellBasedSandbox",
]
