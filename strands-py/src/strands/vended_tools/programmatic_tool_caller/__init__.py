"""Tool for orchestrating an agent's other tools with Python code.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, programmatic_tool_caller

    agent = Agent(tools=[programmatic_tool_caller, bash])
    ```
"""

from .programmatic_tool_caller import make_programmatic_tool_caller, programmatic_tool_caller

__all__ = [
    "make_programmatic_tool_caller",
    "programmatic_tool_caller",
]
