"""Built-in tools for executing commands, editing files, and running code.

The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash`, :func:`make_file_editor`,
and :func:`make_code_execution` factories produce sandbox-routed tools that
either bind to a :class:`~strands.sandbox.base.Sandbox` at creation (as the
built-in Docker/SSH sandboxes do when vending tools) or read the sandbox from
the agent at call time.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, code_execution, file_editor

    agent = Agent(tools=[bash, code_execution, file_editor])
    ```
"""

from .bash import bash, make_bash
from .code_execution import code_execution, make_code_execution
from .file_editor import file_editor, make_file_editor

__all__ = [
    "bash",
    "code_execution",
    "file_editor",
    "make_bash",
    "make_code_execution",
    "make_file_editor",
]
