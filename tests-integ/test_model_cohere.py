import os

import pytest

import strands
from strands import Agent
from strands.models.openai import OpenAIModel


@pytest.fixture
def model():
    return OpenAIModel(
        client_args={
            "base_url": "https://api.cohere.com/compatibility/v1",
            "api_key": os.getenv("CO_API_KEY"),
        },
        model_id="command-a-03-2025",
        params={"stream_options": None},
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
