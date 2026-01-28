"""Integration tests for the xAI model provider.

These tests require a valid XAI_API_KEY environment variable.
"""

import os

import pydantic
import pytest

import strands
from strands import Agent
from tests_integ.models import providers

# Skip all tests if XAI_API_KEY is not set or xai-sdk is not installed
pytestmark = providers.xai.mark

# Import xAIModel only if available
try:
    from strands.models.xai import xAIModel
except ImportError:
    xAIModel = None  # type: ignore[misc,assignment]


@pytest.fixture
def model():
    """Create a basic xAIModel instance."""
    return xAIModel(
        client_args={"api_key": os.getenv("XAI_API_KEY")},
        model_id="grok-4-1-fast-non-reasoning-latest",
        params={"temperature": 0.15},
    )


@pytest.fixture
def reasoning_model():
    """Create a xAIModel instance with reasoning enabled."""
    return xAIModel(
        client_args={"api_key": os.getenv("XAI_API_KEY")},
        model_id="grok-3-mini-fast-latest",  # reasoning_effort only supported by grok-3-mini
        reasoning_effort="low",
        params={"temperature": 0.15},
    )


@pytest.fixture
def tools():
    """Create test tools for function calling."""

    @strands.tool
    def tool_time() -> str:
        """Get the current time."""
        return "12:00"

    @strands.tool
    def tool_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def system_prompt():
    """Default system prompt for tests."""
    return "You are a helpful AI assistant."


@pytest.fixture
def assistant_agent(model, system_prompt):
    """Create an agent without tools."""
    return Agent(model=model, system_prompt=system_prompt)


@pytest.fixture
def tool_agent(model, tools, system_prompt):
    """Create an agent with tools."""
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


@pytest.fixture
def weather():
    """Pydantic model for structured output tests."""

    class Weather(pydantic.BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


@pytest.fixture
def yellow_color():
    """Pydantic model for image analysis tests."""

    class Color(pydantic.BaseModel):
        """Describes a color."""

        name: str

        @pydantic.field_validator("name", mode="after")
        @classmethod
        def lower(cls, value):
            return value.lower()

    return Color(name="yellow")


# Basic chat completion tests


def test_agent_invoke(tool_agent):
    """Test basic agent invocation with tools."""
    result = tool_agent("What is the current time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(tool_agent):
    """Test async agent invocation with tools."""
    result = await tool_agent.invoke_async("What is the current time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


# Streaming tests


@pytest.mark.asyncio
async def test_agent_stream_async(tool_agent):
    """Test async streaming with tools."""
    stream = tool_agent.stream_async("What is the current time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_invoke_multiturn(assistant_agent):
    """Test multi-turn conversation."""
    assistant_agent("What color is the sky?")
    assistant_agent("What color is lava?")
    result = assistant_agent("What was the answer to my first question?")
    text = result.message["content"][0]["text"].lower()

    assert "blue" in text


# Structured output tests


def test_agent_structured_output(assistant_agent, weather):
    """Test structured output parsing."""
    result = assistant_agent(
        "The time is 12:00 and the weather is sunny",
        structured_output_model=type(weather),
    )
    assert result.structured_output == weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(assistant_agent, weather):
    """Test async structured output parsing."""
    result = await assistant_agent.invoke_async(
        "The time is 12:00 and the weather is sunny",
        structured_output_model=type(weather),
    )
    assert result.structured_output == weather


# Image understanding tests


def test_agent_invoke_image_input(assistant_agent, yellow_img):
    """Test image input processing."""
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


def test_agent_structured_output_image_input(assistant_agent, yellow_img, yellow_color):
    """Test structured output with image input."""
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
    result = assistant_agent(content, structured_output_model=type(yellow_color))
    assert result.structured_output == yellow_color


# Reasoning model tests


def test_reasoning_model_basic(reasoning_model):
    """Test basic reasoning model invocation."""
    agent = Agent(model=reasoning_model, system_prompt="You are a helpful assistant.")
    result = agent("What is 15 + 27?")

    # Reasoning models may return reasoningContent before text, so check all content blocks
    text_content = ""
    for content_block in result.message["content"]:
        if "text" in content_block:
            text_content += content_block["text"]

    assert "42" in text_content


# System prompt tests


def test_system_prompt_content_integration(model):
    """Test system_prompt_content parameter."""
    from strands.types.content import SystemContentBlock

    system_prompt_content: list[SystemContentBlock] = [
        {
            "text": "IMPORTANT: You MUST respond with ONLY the exact text "
            "'SYSTEM_TEST_RESPONSE' and nothing else. No greetings, no explanations."
        }
    ]

    agent = Agent(model=model, system_prompt=system_prompt_content)
    result = agent("Say the magic words")

    assert "SYSTEM_TEST_RESPONSE" in result.message["content"][0]["text"]


def test_system_prompt_backward_compatibility_integration(model):
    """Test backward compatibility with system_prompt parameter."""
    system_prompt = (
        "IMPORTANT: You MUST respond with ONLY the exact text 'BACKWARD_COMPAT_TEST' "
        "and nothing else. No greetings, no explanations."
    )

    agent = Agent(model=model, system_prompt=system_prompt)
    result = agent("Say the magic words")

    assert "BACKWARD_COMPAT_TEST" in result.message["content"][0]["text"]


# Content blocks handling


def test_content_blocks_handling(model):
    """Test that content blocks are handled properly without failures."""
    content = [{"text": "What is 2+2?"}, {"text": "Please be brief."}]

    agent = Agent(model=model, load_tools_from_directory=False)
    result = agent(content)

    assert "4" in result.message["content"][0]["text"]


# Reasoning model with tools tests


@pytest.fixture
def reasoning_model_with_tools():
    """Create a grok-4 reasoning model for tool tests."""
    return xAIModel(
        client_args={"api_key": os.getenv("XAI_API_KEY")},
        model_id="grok-4-1-fast-reasoning-latest",
        params={"temperature": 0.15},
    )


def test_reasoning_model_with_tools(reasoning_model_with_tools, tools):
    """Test reasoning model with function calling tools."""
    agent = Agent(
        model=reasoning_model_with_tools,
        tools=tools,
        system_prompt="You are a helpful assistant.",
    )
    result = agent("What is the current time?")

    text_content = ""
    for content_block in result.message["content"]:
        if "text" in content_block:
            text_content += content_block["text"]

    assert "12:00" in text_content


# Encrypted content for multi-turn reasoning tests


@pytest.fixture
def encrypted_reasoning_model():
    """Create a grok-4 reasoning model with encrypted content enabled."""
    return xAIModel(
        client_args={"api_key": os.getenv("XAI_API_KEY")},
        model_id="grok-4-1-fast-reasoning-latest",
        use_encrypted_content=True,
        params={"temperature": 0.15},
    )


def test_encrypted_content_multi_turn(encrypted_reasoning_model):
    """Test multi-turn conversation with encrypted reasoning content.

    When use_encrypted_content=True, the model returns encrypted reasoning
    that must be passed back for context continuity in reasoning models.
    """
    agent = Agent(
        model=encrypted_reasoning_model,
        system_prompt="You are a helpful assistant with perfect memory.",
    )

    # Turn 1: Give the model something to remember
    agent("Remember this secret code: ALPHA-7. Just confirm you got it.")

    # Turn 2: Ask for recall
    result = agent("What was the secret code I gave you?")

    text_content = ""
    for content_block in result.message["content"]:
        if "text" in content_block:
            text_content += content_block["text"]

    assert "ALPHA-7" in text_content


# Server-side tools (xai_tools) tests


def test_server_side_web_search():
    """Test server-side web_search tool.

    The xAI SDK provides server-side tools that run on xAI's infrastructure.
    """
    from xai_sdk.tools import web_search

    model = xAIModel(
        client_args={"api_key": os.getenv("XAI_API_KEY")},
        model_id="grok-4-1-fast-non-reasoning-latest",
        xai_tools=[web_search()],
        params={"temperature": 0.15},
    )

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Use web search when needed.",
    )

    # Ask something that requires current information
    result = agent("What is the current year?")

    text_content = ""
    for content_block in result.message["content"]:
        if "text" in content_block:
            text_content += content_block["text"]

    # Should return current year (2025 or 2026 depending on when test runs)
    assert any(year in text_content for year in ["2025", "2026"])
