"""Bash tool for executing shell commands through a sandbox.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash

    agent = Agent(tools=[bash])
    ```
"""

from .bash import bash, make_bash

__all__ = [
    "bash",
    "make_bash",
]
