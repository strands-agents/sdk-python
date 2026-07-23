"""HTTP request tool for calling external APIs.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import http_request

    agent = Agent(tools=[http_request])
    ```
"""

from .http_request import http_request

__all__ = ["http_request"]
