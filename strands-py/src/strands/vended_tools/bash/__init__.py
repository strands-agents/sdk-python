"""Bash tools for executing shell commands.

Two tools are provided, mirroring ``strands-ts/src/vended-tools/bash/``:

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

from .bash import DEFAULT_BASH_DESCRIPTION, bash, make_bash
from .types import SANDBOX_BASH_DESCRIPTION, BashSessionError, BashTimeoutError

__all__ = [
    "DEFAULT_BASH_DESCRIPTION",
    "SANDBOX_BASH_DESCRIPTION",
    "BashSessionError",
    "BashTimeoutError",
    "bash",
    "make_bash",
]
