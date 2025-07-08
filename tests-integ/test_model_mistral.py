import os

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.models.mistral import MistralModel


@pytest.fixture(scope="module")
def streaming_model():
    return MistralModel(
        model_id="mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        stream=True,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )


@pytest.fixture(scope="module")
def non_streaming_model():
    return MistralModel(
        model_id="mistral-medium-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        stream=False,
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )


@pytest.fixture(scope="module")
def system_prompt():
    return "You are an AI assistant that provides helpful and accurate information."


@pytest.fixture(scope="module")
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture(scope="module")
def streaming_agent(streaming_model, tools):
    return Agent(model=streaming_model, tools=tools)


@pytest.fixture(scope="module")
def non_streaming_agent(non_streaming_model, tools):
    return Agent(model=non_streaming_model, tools=tools)


@pytest.fixture(params=["streaming_agent", "non_streaming_agent"])
def agent(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def weather():
    class Weather(BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_agent_invoke(agent):
    # TODO: https://github.com/strands-agents/sdk-python/issues/374
    # result = streaming_agent("What is the time and weather in New York?")
    result = agent("What is the time in New York?")
    text = result.message["content"][0]["text"].lower()

    # assert all(string in text for string in ["12:00", "sunny"])
    assert all(string in text for string in ["12:00"])


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    # TODO: https://github.com/strands-agents/sdk-python/issues/374
    # result = await streaming_agent.invoke_async("What is the time and weather in New York?")
    result = await agent.invoke_async("What is the time in New York?")
    text = result.message["content"][0]["text"].lower()

    # assert all(string in text for string in ["12:00", "sunny"])
    assert all(string in text for string in ["12:00"])


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    # TODO: https://github.com/strands-agents/sdk-python/issues/374
    # stream = streaming_agent.stream_async("What is the time and weather in New York?")
    stream = agent.stream_async("What is the time in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    # assert all(string in text for string in ["12:00", "sunny"])
    assert all(string in text for string in ["12:00"])


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
def test_agent_structured_output(non_streaming_agent, weather):
    tru_weather = non_streaming_agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.skipif("MISTRAL_API_KEY" not in os.environ, reason="MISTRAL_API_KEY environment variable missing")
@pytest.mark.asyncio
async def test_agent_structured_output_async(non_streaming_agent, weather):
    tru_weather = await non_streaming_agent.structured_output_async(
        type(weather), "The time is 12:00 and the weather is sunny"
    )
    exp_weather = weather
    assert tru_weather == exp_weather
