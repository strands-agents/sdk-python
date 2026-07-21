"""Vended ``use_agent`` tool for delegating tasks to a nested agent.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import use_agent

    agent = Agent(tools=[use_agent])
    ```
"""

from .use_agent import MultiagentDepthExceeded, make_use_agent, use_agent

__all__ = [
    "MultiagentDepthExceeded",
    "make_use_agent",
    "use_agent",
]
