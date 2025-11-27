import os

import pydantic
import pytest

import strands
from strands import Agent
from strands.models.sap_genai_hub import SAPGenAIHubModel
from tests_integ.models import providers

# these tests only run if we have the SAP AI Core credentials
pytestmark = providers.sap_genai_hub.mark


@pytest.fixture
def model():
    return SAPGenAIHubModel(
        model_id="amazon--nova-lite",
        temperature=0.15,  # Lower temperature for consistent test behavior
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
    return "You are a helpful AI assistant."


@pytest.fixture
def assistant_agent(model, system_prompt):
    return Agent(model=model, system_prompt=system_prompt)


@pytest.fixture
def tool_agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


@pytest.fixture
def weather():
    class Weather(pydantic.BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


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


def test_agent_invoke(tool_agent):
    result = tool_agent("What is the current time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(tool_agent):
    result = await tool_agent.invoke_async(
        "What is the current time and weather in New York?"
    )
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(tool_agent):
    stream = tool_agent.stream_async(
        "What is the current time and weather in New York?"
    )
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_invoke_multiturn(assistant_agent):
    assistant_agent("What color is the sky?")
    assistant_agent("What color is lava?")
    result = assistant_agent("What was the answer to my first question?")
    text = result.message["content"][0]["text"].lower()

    assert "blue" in text


def test_agent_invoke_image_input(assistant_agent, yellow_img):
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
    result = assistant_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_agent_invoke_document_input(assistant_agent, letter_pdf):
    content = [
        {"text": "summarize this document"},
        {
            "document": {
                "format": "pdf",
                "name": "letter_name",
                "source": {"bytes": letter_pdf},
            }
        },
    ]
    result = assistant_agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "shareholder" in text


def test_agent_structured_output(assistant_agent, weather):
    tru_weather = assistant_agent.structured_output(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(assistant_agent, weather):
    tru_weather = await assistant_agent.structured_output_async(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather


def test_agent_structured_output_image_input(assistant_agent, yellow_img, yellow_color):
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
    tru_color = assistant_agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic--claude-4.5-sonnet",
        "anthropic--claude-4-sonnet",
        "anthropic--claude-3.7-sonnet",
        "anthropic--claude-3.5-sonnet",
        "anthropic--claude-3-haiku",
        "amazon--nova-pro",
        "amazon--nova-lite",
        "amazon--nova-micro",
    ],
)
def test_different_models(model_id):
    """Test various SAP GenAI Hub models."""
    model = SAPGenAIHubModel(model_id=model_id)
    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Keep responses very brief.",
    )

    result = agent("What is 2+2?")
    text = result.message["content"][0]["text"]

    assert "4" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic--claude-4.5-sonnet",
        "anthropic--claude-4-sonnet",
        "anthropic--claude-3.7-sonnet",
        "anthropic--claude-3.5-sonnet",
        "anthropic--claude-3-haiku",
        "amazon--nova-pro",
        "amazon--nova-lite",
        "amazon--nova-micro",
    ],
)
async def test_streaming_for_models(model_id):
    """Test streaming for various SAP GenAI Hub models."""
    model = SAPGenAIHubModel(model_id=model_id)
    agent = Agent(model=model)

    chunk_count = 0
    stream = agent.stream_async("Count from 1 to 5.")

    async for event in stream:
        if "data" in event and event["data"]:
            chunk_count += 1

    assert chunk_count > 0


def test_multi_agent_workflow():
    """Test multi-agent workflow using agents as tools pattern."""
    from textwrap import dedent

    nova_model = SAPGenAIHubModel(model_id="amazon--nova-lite")
    claude_model = SAPGenAIHubModel(model_id="anthropic--claude-3.5-sonnet")

    @strands.tool
    def research_assistant(query: str) -> str:
        """Research assistant that provides factual information."""
        research_agent = Agent(
            model=claude_model,
            system_prompt=dedent(
                """You are a specialized research assistant. Focus only on providing
                factual information. Keep responses brief and to the point."""
            ),
        )
        return research_agent(query).message

    @strands.tool
    def creative_writing_assistant(query: str) -> str:
        """Creative writing assistant that generates creative content."""
        creative_agent = Agent(
            model=nova_model,
            system_prompt=dedent(
                """You are a specialized creative writing assistant.
                Create engaging content. Keep responses brief and focused."""
            ),
        )
        return creative_agent(query).message

    orchestrator = Agent(
        model=nova_model,
        system_prompt="""You are an assistant that routes queries to specialized agents:
- For research questions use research_assistant
- For creative writing use creative_writing_assistant
- For simple questions answer directly""",
        tools=[research_assistant, creative_writing_assistant],
    )

    result = orchestrator("What is quantum computing? (1 sentence)")
    text = result.message["content"][0]["text"].lower()

    assert "quantum" in text or "computing" in text


def test_model_with_custom_parameters():
    """Test model with custom parameters."""
    model = SAPGenAIHubModel(
        model_id="amazon--nova-lite", temperature=0.3, top_p=0.9, max_tokens=50
    )
    agent = Agent(model=model)

    result = agent("Count from 1 to 5.")

    assert len(result.message["content"][0]["text"]) > 0
