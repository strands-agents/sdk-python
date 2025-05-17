import pytest

import strands
from strands import Agent
from strands.models.sagemaker import SageMakerAIModel

import boto3


@pytest.fixture
def model(endpoint_name: str):
    return SageMakerAIModel(
        endpoint_name=endpoint_name,
        boto_session=boto3.Session(region_name="us-east-1"),
        max_tokens=1024
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


def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])
