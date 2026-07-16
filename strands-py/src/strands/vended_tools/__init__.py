"""Built-in tools for executing commands, editing files, and delegating to nested agents.

The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash` and :func:`make_file_editor`
factories produce sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.

The :data:`use_agent` tool lets the calling agent construct a nested agent
(system prompt, model, and an allowlisted subset of the parent's tools) and
delegate a single task to it.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, file_editor, use_agent

    agent = Agent(tools=[bash, file_editor, use_agent])
    ```
"""

from .bash import bash, make_bash
from .file_editor import file_editor, make_file_editor
from .use_agent import MultiagentDepthExceeded, make_use_agent, use_agent

__all__ = [
    "MultiagentDepthExceeded",
    "bash",
    "file_editor",
    "make_bash",
    "make_file_editor",
    "make_use_agent",
    "use_agent",
]
