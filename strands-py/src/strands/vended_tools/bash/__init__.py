"""Bash tools for executing shell commands.

Two tools are provided:

- :data:`bash` — a host-session tool whose shell state (variables, working
  directory) persists across calls. Runs on the host without sandboxing.
- :func:`make_bash` — a factory for a stateless, sandbox-routed bash tool. Each
  call runs in a fresh shell through the agent's (or a bound) sandbox.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools.bash import bash, make_bash

    # Persistent host session
    agent = Agent(tools=[bash])

    # Stateless, routed through a sandbox
    agent = Agent(tools=[make_bash()])
    ```
"""

from .bash import bash, make_bash

__all__ = [
    "bash",
    "make_bash",
]
