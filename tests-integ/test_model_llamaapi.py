import os

import pytest

import strands
from strands import Agent
from strands.models.llamaapi import LlamaAPIModel


@pytest.fixture
def model():
    return LlamaAPIModel(
        model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
        client_args={
            "api_key": os.getenv("LLAMA_API_KEY"),
        },
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
    "LLAMA_API_KEY" not in os.environ,
    reason="LLAMA_API_KEY environment variable missing",
)
def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])
