"""Built-in tools for executing commands, editing files, and controlling the agent loop.

The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash` and :func:`make_file_editor`
factories produce sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.
The :data:`stop` tool signals the event loop to end the current invocation
gracefully.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, file_editor, stop

    agent = Agent(tools=[bash, file_editor, stop])
    ```
"""

from .bash import bash, make_bash
from .file_editor import file_editor, make_file_editor
from .stop import make_stop, stop

__all__ = [
    "bash",
    "file_editor",
    "make_bash",
    "make_file_editor",
    "make_stop",
    "stop",
]
