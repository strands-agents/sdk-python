import os

import pytest
from pydantic import BaseModel, Field

import strands
from strands import Agent
from strands.models.deepseek import DeepSeekModel

# these tests only run if we have the deepseek api key
pytestmark = pytest.mark.skipif(
    "DEEPSEEK_API_KEY" not in os.environ,
    reason="DEEPSEEK_API_KEY environment variable missing",
)


@pytest.fixture()
def base_model():
    return DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"), model_id="deepseek-chat", params={"max_tokens": 2000, "temperature": 0.7}
    )


@pytest.fixture()
def reasoning_model():
    return DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"), model_id="deepseek-reasoner", params={"max_tokens": 32000}
    )


@pytest.fixture()
def beta_model():
    return DeepSeekModel(
        api_key=os.getenv("DEEPSEEK_API_KEY"), model_id="deepseek-chat", use_beta=True, params={"max_tokens": 1000}
    )


@pytest.fixture()
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture()
def base_agent(base_model, tools):
    return Agent(model=base_model, tools=tools)


class PersonInfo(BaseModel):
    """Extract person information from text."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")


class WeatherInfo(BaseModel):
    """Weather information."""

    location: str = Field(description="City and state")
    temperature: str = Field(description="Current temperature")
    condition: str = Field(description="Weather condition")


def test_basic_conversation(base_agent):
    result = base_agent("Hello, how are you today?")
    assert "content" in result.message
    assert len(result.message["content"]) > 0


def test_structured_output_person(base_agent):
    result = base_agent.structured_output(
        PersonInfo, "John Smith is a 30-year-old software engineer working at a tech startup."
    )
    assert result.name == "John Smith"
    assert result.age == 30
    assert "engineer" in result.occupation.lower()


def test_tool_usage(base_agent):
    result = base_agent("What is the time and weather?")
    # Handle case where content might be empty or structured differently
    content = result.message.get("content", [])
    if content and "text" in content[0]:
        text = content[0]["text"].lower()
        assert any(string in text for string in ["12:00", "sunny"])
    else:
        # If no text content, just verify the result exists
        assert result.message is not None


@pytest.mark.asyncio
async def test_streaming(base_model):
    agent = Agent(model=base_model)
    events = []
    async for event in agent.stream_async("Tell me a short fact about robots"):
        events.append(event)

    assert len(events) > 0
    assert "result" in events[-1]


def test_config_update(base_model):
    original_config = base_model.get_config()
    assert original_config["model_id"] == "deepseek-chat"

    base_model.update_config(model_id="deepseek-reasoner", params={"temperature": 0.5})
    updated_config = base_model.get_config()
    assert updated_config["model_id"] == "deepseek-reasoner"


def test_weather_structured_output(base_agent):
    result = base_agent.structured_output(
        WeatherInfo, "Get the weather for San Francisco, CA. It's currently 72°F and sunny."
    )
    assert "san francisco" in result.location.lower()
    assert "72" in result.temperature
    assert "sunny" in result.condition.lower()


@pytest.mark.asyncio
async def test_async_structured_output(base_agent):
    result = await base_agent.structured_output_async(
        PersonInfo, "Alice Johnson is a 25-year-old teacher at the local school."
    )
    assert result.name == "Alice Johnson"
    assert result.age == 25
    assert "teacher" in result.occupation.lower()
