"""Tool for gracefully ending the agent loop.

This tool is experimental and subject to change in future revisions without notice.

Example Usage:
    ```python
    from strands import Agent
    from strands.experimental.tools import stop

    agent = Agent(tools=[stop])
    ```
"""

from .stop import make_stop, stop

__all__ = [
    "make_stop",
    "stop",
]
