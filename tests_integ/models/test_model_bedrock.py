import pydantic
import pytest

import strands
from strands import Agent
from strands.models import BedrockModel
from strands.types.content import ContentBlock


@pytest.fixture
def system_prompt():
    return "You are an AI assistant that uses & instead of ."


@pytest.fixture
def streaming_model():
    return BedrockModel(
        streaming=True,
    )


@pytest.fixture
def non_streaming_model():
    return BedrockModel(
        streaming=False,
    )


@pytest.fixture
def streaming_agent(streaming_model, system_prompt):
    return Agent(
        model=streaming_model,
        system_prompt=system_prompt,
        load_tools_from_directory=False,
    )


@pytest.fixture
def non_streaming_agent(non_streaming_model, system_prompt):
    return Agent(
        model=non_streaming_model,
        system_prompt=system_prompt,
        load_tools_from_directory=False,
    )


@pytest.fixture
def yellow_color():
    class Color(pydantic.BaseModel):
        """Describes a color."""

        name: str

        @pydantic.field_validator("name", mode="after")
        @classmethod
        def lower(_, value):
            return value.lower()

    return Color(name="yellow")


def test_streaming_agent(streaming_agent):
    """Test agent with streaming model."""
    result = streaming_agent("Hello!")

    assert len(str(result)) > 0


def test_non_streaming_agent(non_streaming_agent):
    """Test agent with non-streaming model."""
    result = non_streaming_agent("Hello!")

    assert len(str(result)) > 0


@pytest.mark.asyncio
async def test_streaming_model_events(streaming_model, alist):
    """Test streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call stream and collect events
    events = await alist(streaming_model.stream(messages))

    # Verify basic structure of events
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)
    assert any("messageStop" in event for event in events)


@pytest.mark.asyncio
async def test_non_streaming_model_events(non_streaming_model, alist):
    """Test non-streaming model events."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    # Call stream and collect events
    events = await alist(non_streaming_model.stream(messages))

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

    class Weather(pydantic.BaseModel):
        time: str
        weather: str

    agent = Agent(model=streaming_model)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


def test_structured_output_non_streaming(non_streaming_model):
    """Test structured output with non-streaming model."""

    class Weather(pydantic.BaseModel):
        time: str
        weather: str

    agent = Agent(model=non_streaming_model)

    result = agent.structured_output(Weather, "The time is 12:00 and the weather is sunny")
    assert isinstance(result, Weather)
    assert result.time == "12:00"
    assert result.weather == "sunny"


def test_invoke_multi_modal_input(streaming_agent, yellow_img):
    content = [
        {"text": "what is in this image"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    result = streaming_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_document_citations(non_streaming_agent, letter_pdf):
    content: list[ContentBlock] = [
        {
            "document": {
                "name": "letter to shareholders",
                "source": {"bytes": letter_pdf},
                "citations": {"enabled": True},
                "context": "This is a letter to shareholders",
                "format": "pdf",
            },
        },
        {"text": "What does the document say about artificial intelligence? Use citations to back up your answer."},
    ]
    non_streaming_agent(content)

    assert any("citationsContent" in content for content in non_streaming_agent.messages[-1]["content"])


def test_document_citations_streaming(streaming_agent, letter_pdf):
    content: list[ContentBlock] = [
        {
            "document": {
                "name": "letter to shareholders",
                "source": {"bytes": letter_pdf},
                "citations": {"enabled": True},
                "context": "This is a letter to shareholders",
                "format": "pdf",
            },
        },
        {"text": "What does the document say about artificial intelligence? Use citations to back up your answer."},
    ]
    streaming_agent(content)

    assert any("citationsContent" in content for content in streaming_agent.messages[-1]["content"])


def test_structured_output_multi_modal_input(streaming_agent, yellow_img, yellow_color):
    content = [
        {"text": "Is this image red, blue, or yellow?"},
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": yellow_img,
                },
            },
        },
    ]
    tru_color = streaming_agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color
