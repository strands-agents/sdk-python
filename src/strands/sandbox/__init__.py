"""Sandbox abstraction for agent code execution environments.

This module provides the Sandbox interface that decouples tool logic from where code runs.
Tools that need to execute code or access a filesystem receive a Sandbox instead of managing
their own execution, enabling portability across local, Docker, and cloud environments.

Class hierarchy::

    Sandbox (ABC, all 5 abstract + lifecycle)
      └── ShellBasedSandbox (ABC, only execute() abstract — shell-based file ops + execute_code)
            ├── LocalSandbox — runs on the host via asyncio subprocesses (default)
            └── DockerSandbox — runs inside a Docker container
"""

from .base import ExecutionResult, Sandbox, ShellBasedSandbox
from .docker import DockerSandbox
from .local import LocalSandbox

__all__ = [
    "DockerSandbox",
    "ExecutionResult",
    "LocalSandbox",
    "Sandbox",
    "ShellBasedSandbox",
]
