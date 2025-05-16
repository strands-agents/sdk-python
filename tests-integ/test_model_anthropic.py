import os

import pytest

import strands
from strands import Agent
from strands.models.anthropic import AnthropicModel


@pytest.fixture
def model():
    return AnthropicModel(
        client_args={
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        model_id="claude-3-7-sonnet-20250219",
        max_tokens=512,
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
def system_prompt():
    return "You are an AI assistant that uses & instead of ."


@pytest.fixture
def agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


@pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY environment variable missing")
def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny", "&"])
