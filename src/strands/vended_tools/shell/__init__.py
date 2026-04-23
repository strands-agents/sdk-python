"""Shell tool for executing commands in the agent's sandbox.

Example::

    from strands import Agent
    from strands.vended_tools import shell

    agent = Agent(tools=[shell])
    agent("List all Python files in the current directory")

Configuration via agent state::

    agent.state.set("strands_shell_tool", {
        "timeout": 120,  # Default timeout in seconds
    })
"""

from .shell import shell

__all__ = ["shell"]
