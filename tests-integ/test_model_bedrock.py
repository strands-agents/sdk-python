import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.models import BedrockModel


@pytest.fixture
def system_prompt():
    return "You are an AI assistant that uses & instead of ."


@pytest.fixture
def streaming_model():
    return BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        streaming=True,
    )


@pytest.fixture
def non_streaming_model():
    return BedrockModel(
        model_id="us.meta.llama3-2-90b-instruct-v1:0",
        streaming=False,
    )


@pytest.fixture
def streaming_agent(streaming_model, system_prompt):
    return Agent(model=streaming_model, system_prompt=system_prompt, load_tools_from_directory=False)


@pytest.fixture
def non_streaming_agent(non_streaming_model, system_prompt):
    return Agent(model=non_streaming_model, system_prompt=system_prompt, load_tools_from_directory=False)


def test_streaming_agent(streaming_agent):
    """Test agent with streaming model."""
    result = streaming_agent("Hello!")

    assert len(str(result)) > 0


def test_non_streaming_agent(non_streaming_agent):
    """Test agent with non-streaming model."""
    result = non_streaming_agent("Hello!")

    assert len(str(result)) > 0


@pytest.mark.asyncio
async def test_streaming_model_events(streaming_model):
    """Test streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call converse and collect events
    events = [event async for event in streaming_model.converse(messages)]

    # Verify basic structure of events
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)
    assert any("messageStop" in event for event in events)


@pytest.mark.asyncio
async def test_non_streaming_model_events(non_streaming_model):
    """Test non-streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call converse and collect events
    events = [event async for event in non_streaming_model.converse(messages)]

    # Verify basic structure of events
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)
    assert any("messageStop" in event for event in events)


def test_tool_use_streaming(streaming_model):
    """Test tool use with streaming model."""

    tool_was_called = False

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""

        nonlocal tool_was_called
        tool_was_called = True
        return eval(expression)

    agent = Agent(model=streaming_model, tools=[calculator], load_tools_from_directory=False)
    result = agent("What is 123 + 456?")

    # Print the full message content for debugging
    print("\nFull message content:")
    import json

    print(json.dumps(result.message["content"], indent=2))

    assert tool_was_called


def test_tool_use_non_streaming(non_streaming_model):
    """Test tool use with non-streaming model."""

    tool_was_called = False

    @strands.tool
    def calculator(expression: str) -> float:
        """Calculate the result of a mathematical expression."""

        nonlocal tool_was_called
        tool_was_called = True
        return eval(expression)

    agent = Agent(model=non_streaming_model, tools=[calculator], load_tools_from_directory=False)
    agent("What is 123 + 456?")

    assert tool_was_called


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
