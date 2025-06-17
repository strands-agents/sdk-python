import os

import pytest

import strands
from strands import Agent
from strands.models.cohere import CohereModel


@pytest.fixture
def model():
    return CohereModel(
        model_id="command-a-03-2025",
        api_key=os.getenv("CO_API_KEY"),
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools):
    return Agent(model=model, tools=tools)


@pytest.mark.skipif(
    "CO_API_KEY" not in os.environ,
    reason="CO_API_KEY environment variable missing",
)
def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()
    assert all(string in text for string in ["12:00", "sunny"])
