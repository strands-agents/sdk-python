import os

import pydantic
import pytest

import strands
from strands import Agent, tool
from strands.models.openai import OpenAIModel
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
