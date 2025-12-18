import os

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.models.neuronvllm import NeuronVLLMModel
from tests_integ.models import providers

# these tests only run if we have the neuron-vLLM server running
pytestmark = providers.neuronvllm.mark


@pytest.fixture
def model() -> NeuronVLLMModel:
    base_url = os.getenv("OPENAI_API_BASE_URL", "http://localhost:8084/v1")
    model_id = os.getenv("NEURON_VLLM_MODEL_ID", "neuron-llama-3.1-70b-instruct")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

    return NeuronVLLMModel(
        {
            "model_id": model_id,
            "openai_api_base": base_url,
            "openai_api_key": api_key,
        }
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
def agent(model: NeuronVLLMModel, tools):
    return Agent(model=model, tools=tools)


@pytest.fixture
def weather():
    class Weather(BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


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
    tru_weather = await agent.structured_output_async(
        type(weather),
        "The time is 12:00 and the weather is sunny",
    )
    exp_weather = weather
    assert tru_weather == exp_weather


