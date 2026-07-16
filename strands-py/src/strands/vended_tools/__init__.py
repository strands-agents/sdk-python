"""Built-in tools for executing commands and editing files.

The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash` and :func:`make_file_editor`
factories produce sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.

:data:`a2a_client` and :func:`make_a2a_client` provide a thin shim over the
optional :class:`~strands.agent.a2a_agent.A2AAgent` client for invoking remote
A2A agents by URL.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, file_editor

    agent = Agent(tools=[bash, file_editor])
    ```
"""

from .a2a_client import a2a_client, make_a2a_client
from .bash import bash, make_bash
from .file_editor import file_editor, make_file_editor

__all__ = [
    "a2a_client",
    "bash",
    "file_editor",
    "make_a2a_client",
    "make_bash",
    "make_file_editor",
]
