import os

import pytest

import strands
from strands import Agent
from strands.models.sagemaker import SageMakerAIModel


@pytest.fixture
def model():
    endpoint_config = SageMakerAIModel.SageMakerAIEndpointConfig(
        endpoint_name=os.getenv("SAGEMAKER_ENDPOINT_NAME", "mistral-small-2501-sm-js"), region_name="us-east-1"
    )
    payload_config = SageMakerAIModel.SageMakerAIPayloadSchema(max_tokens=1024, temperature=0.7, stream=False)
    return SageMakerAIModel(endpoint_config=endpoint_config, payload_config=payload_config)


@pytest.fixture
def tools():
    @strands.tool
    def tool_time(location: str) -> str:
        """Get the current time for a location."""
        return "12:00"

    @strands.tool
    def tool_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant that provides concise answers."


@pytest.fixture
def agent(model, tools, system_prompt):
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


@pytest.mark.skipif(
    "SAGEMAKER_ENDPOINT_NAME" not in os.environ,
    reason="SAGEMAKER_ENDPOINT_NAME environment variable missing",
)
def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert any(string in text for string in ["12:00", "sunny"])