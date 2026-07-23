"""Python snippets for the vended tools guide."""

from strands import Agent
from strands.vended_tools import http_request


def http_request_example() -> None:
    """Show an agent using the HTTP request tool."""
    # --8<-- [start:http_request_example]
    agent = Agent(tools=[http_request])

    agent("Get data from https://api.example.com/users")
    agent('Post {"name": "John"} to https://api.example.com/users')
    # --8<-- [end:http_request_example]
