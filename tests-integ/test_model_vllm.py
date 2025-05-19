import pytest
import strands
from strands import Agent
from strands.models.vllm import VLLMModel


@pytest.fixture
def model():
    return VLLMModel(
        model_id="meta-llama/Llama-3.2-3B",  # or whatever your model ID is
        host="http://localhost:8000",  # adjust as needed
        max_tokens=128,
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "cloudy"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools):
    return Agent(model=model, tools=tools)


def test_agent(agent):
    result = agent("What is the time and weather in Melboune Australia?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["3:00", "cloudy"])
