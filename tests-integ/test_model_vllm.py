import pytest
import strands
from strands import Agent
from strands.models.vllm import VLLMModel


@pytest.fixture
def model():
    return VLLMModel(
        model_id="Qwen/Qwen3-4B",
        host="http://localhost:8000",
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
    # Send prompt
    result = agent("What is the time and weather in Melbourne Australia?")

    # Extract plain text from the first content block
    text_blocks = result.message.get("content", [])
    # content is a list of dicts with 'text' keys
    text = " ".join(block.get("text", "") for block in text_blocks).lower()

    # Assert that the tool outputs appear in the generated response text
    assert "tool_weather" in text
    #assert "cloudy" in text
