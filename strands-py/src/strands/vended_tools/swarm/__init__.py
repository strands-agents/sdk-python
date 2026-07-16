"""Swarm vended tool.

Spin up a handoff-based team of sub-agents at runtime and run them to a final
result. Shims over :class:`~strands.multiagent.swarm.Swarm`.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import swarm

    agent = Agent(tools=[swarm])
    ```
"""

from .swarm import SWARM_TOOL_DESCRIPTION, MultiagentDepthExceeded, make_swarm, swarm

__all__ = [
    "SWARM_TOOL_DESCRIPTION",
    "MultiagentDepthExceeded",
    "make_swarm",
    "swarm",
]
