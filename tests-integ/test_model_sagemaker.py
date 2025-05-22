import pytest

import strands
from strands import Agent
from strands.models.sagemaker import SageMakerAIModel

ENDPOINT_NAME = "mistral-small-2501-sm-js"
REGION_NAME = "us-east-1"


@pytest.fixture
def model():
    return SageMakerAIModel(endpoint_name=ENDPOINT_NAME, region_name=REGION_NAME, max_tokens=1024)


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


def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])
