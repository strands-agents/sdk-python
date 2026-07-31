"""Shell tool for executing commands through a sandbox.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import shell

    agent = Agent(tools=[shell])
    ```
"""

from .shell import make_shell, shell
from .types import ShellExecutionError

__all__ = [
    "ShellExecutionError",
    "make_shell",
    "shell",
]
