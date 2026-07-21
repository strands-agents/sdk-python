"""A2A client tool for invoking remote A2A agents.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools.a2a_client import a2a_client

    agent = Agent(tools=[a2a_client])
    ```
"""

from .a2a_client import a2a_client, make_a2a_client
from .types import A2AClientOutput, A2AClientRemoteCard
from .url_guard import UrlNotAllowedError

__all__ = [
    "A2AClientOutput",
    "A2AClientRemoteCard",
    "UrlNotAllowedError",
    "a2a_client",
    "make_a2a_client",
]
