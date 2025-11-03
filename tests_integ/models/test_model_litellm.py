import unittest.mock

import pydantic
import pytest

import strands
from strands import Agent
from strands.models.litellm import LiteLLMModel


@pytest.fixture
def model():
    return LiteLLMModel(model_id="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0")


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

        time: str = pydantic.Field(description="The time in HH:MM format (e.g., '12:00', '09:30')")
        weather: str = pydantic.Field(description="The weather condition (e.g., 'sunny', 'rainy', 'cloudy')")

    return Weather(time="12:00", weather="sunny")


class Location(pydantic.BaseModel):
    """Location information."""

    city: str = pydantic.Field(description="The city name")
    country: str = pydantic.Field(description="The country name")


class WeatherCondition(pydantic.BaseModel):
    """Weather condition details."""

    condition: str = pydantic.Field(description="The weather condition (e.g., 'sunny', 'rainy', 'cloudy')")
    temperature: int = pydantic.Field(description="Temperature in Celsius")


class NestedWeather(pydantic.BaseModel):
    """Weather report with nested location and condition information."""

    time: str = pydantic.Field(description="The time in HH:MM format")
    location: Location = pydantic.Field(description="Location information")
    weather: WeatherCondition = pydantic.Field(description="Weather condition details")


@pytest.fixture
def nested_weather():
    return NestedWeather(
        time="12:00",
        location=Location(city="New York", country="USA"),
        weather=WeatherCondition(condition="sunny", temperature=25),
    )


@pytest.fixture
def yellow_color():
    class Color(pydantic.BaseModel):
        """Describes a color with its basic name.

        Used to extract and normalize color names from text or images.
        The color name should be a simple, common color like 'red', 'blue', 'yellow', etc.
        """

        simple_color_name: str = pydantic.Field(
            description="The basic color name (e.g., 'red', 'blue', 'yellow', 'green', 'orange', 'purple')"
        )

        @pydantic.field_validator("simple_color_name", mode="after")
        @classmethod
        def lower(_, value):
            return value.lower()

    return Color(simple_color_name="yellow")


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


def test_agent_invoke_reasoning(agent, model):
    model.update_config(
        params={
            "thinking": {
                "budget_tokens": 1024,
                "type": "enabled",
            },
        },
    )

    result = agent("Please reason about the equation 2+2.")

    assert "reasoningContent" in result.message["content"][0]
    assert result.message["content"][0]["reasoningContent"]["reasoningText"]["text"]


def test_structured_output(agent, weather):
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
    result = agent(content)
    text = result.message["content"][0]["text"].lower()

    assert "yellow" in text


def test_structured_output_multi_modal_input(agent, yellow_img, yellow_color):
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
    tru_color = agent.structured_output(type(yellow_color), content)
    exp_color = yellow_color
    assert tru_color == exp_color


def test_structured_output_unsupported_model(model, nested_weather):
    # Mock supports_response_schema to return False to test fallback mechanism
    with (
        unittest.mock.patch.multiple(
            "strands.models.litellm",
            supports_response_schema=unittest.mock.DEFAULT,
        ) as mocks,
        unittest.mock.patch.object(
            model, "_structured_output_using_tool", wraps=model._structured_output_using_tool
        ) as mock_tool,
        unittest.mock.patch.object(
            model, "_structured_output_using_response_schema", wraps=model._structured_output_using_response_schema
        ) as mock_schema,
    ):
        mocks["supports_response_schema"].return_value = False

        # Test that structured output still works via tool calling fallback
        agent = Agent(model=model)
        prompt = "The time is 12:00 in New York, USA and the weather is sunny with temperature 25 degrees Celsius"
        tru_weather = agent.structured_output(NestedWeather, prompt)
        exp_weather = nested_weather
        assert tru_weather == exp_weather

        # Verify that the tool method was called and schema method was not
        mock_tool.assert_called_once()
        mock_schema.assert_not_called()
