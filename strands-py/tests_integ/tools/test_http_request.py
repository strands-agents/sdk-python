"""Integration tests for the HTTP request tool."""

from strands import Agent
from strands.vended_tools import http_request
from tests_integ.models.providers import bedrock


def test_agent_uses_http_request_to_fetch_weather():
    """An agent can use the tool against a live public API."""
    agent = Agent(model=bedrock.create_model(), tools=[http_request])

    result = agent(
        "Use http_request to GET current NYC weather from "
        "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current=temperature_2m. "
        "Then briefly report the temperature."
    )

    assert any(term in str(result).lower() for term in ("weather", "temperature", "°", "nyc", "new york"))
    assert result.stop_reason == "end_turn"
    assert result.message["role"] == "assistant"
