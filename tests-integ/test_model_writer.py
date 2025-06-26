import os

import pytest

import strands
from strands import Agent
from strands.models.writer import WriterModel


@pytest.fixture
def model():
    return WriterModel(
        model="palmyra-x4",
        client_args={"api_key": os.getenv("WRITER_API_KEY", "")},
        stream_options={"include_usage": True},
    )


@pytest.fixture
def system_prompt():
    return "You are a smart assistant, that uses @ instead of all punctuation marks"


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
def agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt, load_tools_from_directory=False)


@pytest.mark.skipif("WRITER_API_KEY" not in os.environ, reason="WRITER_API_KEY environment variable missing")
def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])
