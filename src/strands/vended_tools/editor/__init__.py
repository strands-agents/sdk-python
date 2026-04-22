"""File editor tool for viewing, creating, and editing files in the agent's sandbox.

Example::

    from strands import Agent
    from strands.vended_tools import editor

    agent = Agent(tools=[editor])
    agent("View the contents of /tmp/example.py")

Configuration via agent state::

    agent.state.set("strands_editor_tool", {
        "max_file_size": 1048576,  # Maximum file size in bytes (default: 1MB)
    })
"""

from .editor import editor

__all__ = ["editor"]
