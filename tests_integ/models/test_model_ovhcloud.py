"""Integration tests for OVHcloud AI Endpoints model provider.

These tests require either:
- No API key (free tier with rate limits), or
- OVHCLOUD_API_KEY environment variable set

To run these tests:
1. Optionally set OVHCLOUD_API_KEY environment variable
2. Run: pytest tests_integ/models/test_model_ovhcloud.py

For a list of available models, see:
https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/
"""

import os

import pytest

import strands
from strands import Agent
from strands.models.ovhcloud import OVHcloudModel
from tests_integ.models import providers

# These tests run with or without API key (free tier supported)
pytestmark = providers.ovhcloud.mark


@pytest.fixture
def model():
    """Create an OVHcloud model instance."""
    return OVHcloudModel(
        client_args={
            "api_key": os.getenv("OVHCLOUD_API_KEY") or "",  # Empty string for free tier
        },
        model_id="gpt-oss-120b",
    )


@pytest.fixture
def tools():
    """Create test tools."""

    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools):
    """Create an agent with the model and tools."""
    return Agent(model=model, tools=tools)


def test_agent_invoke(agent):
    """Test basic agent invocation."""
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    """Test async agent invocation."""
    result = await agent.invoke_async("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    """Test async streaming."""
    stream = agent.stream_async("What is the time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])


def test_model_without_api_key():
    """Test that the model works without an API key (free tier)."""
    model = OVHcloudModel(
        client_args={},
        model_id="gpt-oss-20b",
    )
    agent = Agent(model=model)
    result = agent("Say hello in one word")
    assert len(result.message["content"]) > 0
    assert "hello" in result.message["content"][0]["text"].lower()


def test_model_with_empty_string_api_key():
    """Test that the model works with empty string API key (free tier)."""
    model = OVHcloudModel(
        client_args={"api_key": ""},
        model_id="gpt-oss-20b",
    )
    agent = Agent(model=model)
    result = agent("Say hello in one word")
    assert len(result.message["content"]) > 0
    assert "hello" in result.message["content"][0]["text"].lower()
