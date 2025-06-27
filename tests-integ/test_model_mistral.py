import os

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.models.mistral import MistralModel


@pytest.fixture
def streaming_model():
    return MistralModel(
        model_id="mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        stream=True,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )


@pytest.fixture
def non_streaming_model():
    return MistralModel(
        model_id="mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        stream=False,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )


@pytest.fixture
def system_prompt():
    return "You are an AI assistant that provides helpful and accurate information."


@pytest.fixture
def calculator_tool():
    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""
        return eval(expression)

    return calculator


@pytest.fixture
def weather_tools():
    @strands.tool
    def tool_time() -> str:
        """Get the current time."""
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        """Get the current weather."""
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def streaming_agent(streaming_model):
    return Agent(model=streaming_model)


@pytest.fixture
def non_streaming_agent(non_streaming_model):
    return Agent(model=non_streaming_model)


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_streaming_agent_basic(streaming_agent):
    """Test basic streaming agent functionality."""
    result = streaming_agent("Tell me about Agentic AI in one sentence.")

    assert len(str(result)) > 0
    assert hasattr(result, "message")
    assert "content" in result.message


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_non_streaming_agent_basic(non_streaming_agent):
    """Test basic non-streaming agent functionality."""
    result = non_streaming_agent("Tell me about Agentic AI in one sentence.")

    assert len(str(result)) > 0
    assert hasattr(result, "message")
    assert "content" in result.message


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_tool_use_streaming(streaming_model):
    """Test tool use with streaming model."""

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""
        return eval(expression)

    agent = Agent(model=streaming_model, tools=[calculator])
    result = agent("What is the square root of 1764")

    # Verify the result contains the calculation
    text_content = str(result).lower()
    assert "42" in text_content


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_tool_use_non_streaming(non_streaming_model):
    """Test tool use with non-streaming model."""

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""
        return eval(expression)

    agent = Agent(model=non_streaming_model, tools=[calculator], load_tools_from_directory=False)
    result = agent("What is the square root of 1764")

    text_content = str(result).lower()
    assert "42" in text_content


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_structured_output_streaming(streaming_model):
    """Test structured output with streaming model."""

    class Weather(BaseModel):
        time: str
        weather: str

    agent = Agent(model=streaming_model)
    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")

    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_structured_output_non_streaming(non_streaming_model):
    """Test structured output with non-streaming model."""

    class Weather(BaseModel):
        time: str
        weather: str

    agent = Agent(model=non_streaming_model)
    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")

    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"
