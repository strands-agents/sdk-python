import os
import unittest.mock

import pydantic
import pytest

import strands
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from tests_integ.models import providers

# these tests only run if we have the openai api key
pytestmark = providers.openai.mark


@pytest.fixture
def model():
    return OpenAIModel(
        model_id="gpt-4o",
        client_args={
            "api_key": os.getenv("OPENAI_API_KEY"),
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


@pytest.fixture(scope="module")
def test_image_path(request):
    return request.config.rootpath / "tests_integ" / "test_image.png"


def test_agent_invoke(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    result = await agent.invoke_async("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    stream = agent.stream_async("What is the time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_structured_output(agent, weather):
    tru_weather = agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(agent, weather):
    tru_weather = await agent.structured_output_async(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


def test_invoke_multi_modal_input(agent, yellow_img):
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
    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_structured_output_multi_modal_input(agent, yellow_img, yellow_color):
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
    tru_color = agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


@pytest.mark.skip("https://github.com/strands-agents/sdk-python/issues/320")
def test_tool_returning_images(model, yellow_img):
    @tool
    def tool_with_image_return():
        return {
            "status": "success",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": yellow_img},
                    }
                },
            ],
        }

    agent = Agent(model, tools=[tool_with_image_return])
    # NOTE - this currently fails with: "Invalid 'messages[3]'. Image URLs are only allowed for messages with role
    # 'user', but this message with role 'tool' contains an image URL."
    # See https://github.com/strands-agents/sdk-python/issues/320 for additional details
    agent("Run the the tool and analyze the image")


def test_context_window_overflow_integration():
    """Integration test for context window overflow with OpenAI.

    This test verifies that when a request exceeds the model's context window,
    the OpenAI model properly raises a ContextWindowOverflowException.
    """
    # Use gpt-4o-mini which has a smaller context window to make this test more reliable
    mini_model = OpenAIModel(
        model_id="gpt-4o-mini-2024-07-18",
        client_args={
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
    )

    agent = Agent(model=mini_model)

    # Create a very long text that should exceed context window
    # This text is designed to be long enough to exceed context but not hit token rate limits
    long_text = (
        "This text is longer than context window, but short enough to not get caught in token rate limit. " * 6800
    )

    # This should raise ContextWindowOverflowException which gets handled by conversation manager
    # The agent should attempt to reduce context and retry
    with pytest.raises(ContextWindowOverflowException):
        agent(long_text)


def test_rate_limit_throttling_integration_no_retries(model):
    """Integration test for rate limit handling with retries disabled.

    This test verifies that when a request exceeds OpenAI's rate limits,
    the model properly raises a ModelThrottledException. We disable retries
    to avoid waiting for the exponential backoff during testing.
    """
    # Patch the event loop constants to disable retries for this test
    with unittest.mock.patch("strands.event_loop.event_loop.MAX_ATTEMPTS", 1):
        agent = Agent(model=model)

        # Create a message that's very long to trigger token-per-minute rate limits
        # This should be large enough to exceed TPM limits immediately
        very_long_text = "Really long text " * 20000

        # This should raise ModelThrottledException without retries
        with pytest.raises(ModelThrottledException) as exc_info:
            agent(very_long_text)

        # Verify it's a rate limit error
        error_message = str(exc_info.value).lower()
        assert "rate limit" in error_message or "tokens per min" in error_message
