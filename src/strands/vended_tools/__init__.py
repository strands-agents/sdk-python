"""Vended tools for Strands agents.

These are production-ready tools that ship with the SDK and integrate
with the :class:`~strands.sandbox.base.Sandbox` abstraction. They work
transparently whether the agent uses a local :class:`~strands.sandbox.host.HostSandbox`
or a remote sandbox implementation.

Each tool reads its configuration from ``tool_context.agent.state`` using
a namespaced key (e.g., ``strands_shell_tool``). This means configuration
persists across tool calls and survives session serialization.

Available tools:

- :func:`~strands.vended_tools.shell.shell` — Execute shell commands
- :func:`~strands.vended_tools.editor.editor` — View, create, and edit files
- :func:`~strands.vended_tools.python_repl.python_repl` — Execute Python code

Example::

    from strands import Agent
    from strands.vended_tools import shell, editor, python_repl

    agent = Agent(tools=[shell, editor, python_repl])

    # Configure tools via agent state (persists across calls)
    agent.state.set("strands_shell_tool", {
        "timeout": 60,
    })
"""

from .editor import editor
from .python_repl import python_repl
from .shell import shell

__all__ = [
    "editor",
    "python_repl",
    "shell",
]
