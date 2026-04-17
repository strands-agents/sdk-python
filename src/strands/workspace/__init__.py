"""Workspace abstraction for agent code execution environments.

This module provides the Workspace interface that decouples tool logic from where code runs.
Tools that need to execute code or access a filesystem receive a Workspace instead of managing
their own execution, enabling portability across local and cloud environments.

Class hierarchy::

    Workspace (ABC, all 6 abstract + lifecycle + helpers)
      ├── LocalWorkspace — native Python methods for host execution (default)
      └── ShellBasedWorkspace (ABC, only execute() abstract — shell-based file ops + execute_code)
"""

from .base import ExecutionResult, FileInfo, OutputFile, Workspace
from .local import LocalWorkspace
from .shell_based import ShellBasedWorkspace

__all__ = [
    "ExecutionResult",
    "FileInfo",
    "LocalWorkspace",
    "OutputFile",
    "ShellBasedWorkspace",
    "Workspace",
]
