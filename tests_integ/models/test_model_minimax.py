import os

import pydantic
import pytest

import strands
from strands import Agent
from strands.models.minimax import MinimaxModel
from tests_integ.models import providers

# these tests only run if we have the minimax api key
pytestmark = providers.minimax.mark


@pytest.fixture
def model():
    return MinimaxModel(
        model_id="MiniMax-M2.7",
        client_args={
            "api_key": os.getenv("MINIMAX_API_KEY"),
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
        """Extract time and weather values."""

        time: str = pydantic.Field(description="The time value only, e.g. '14:30' not 'The time is 14:30'")
        weather: str = pydantic.Field(
            description="The weather condition only, e.g. 'rainy' not 'the weather is rainy'"
        )

    return Weather(time="12:00", weather="sunny")


def test_agent_invoke(agent, model):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(agent, model):
    result = await agent.invoke_async("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_agent_structured_output(model, weather):
    agent = Agent(model=model)
    result = agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    assert result == weather
