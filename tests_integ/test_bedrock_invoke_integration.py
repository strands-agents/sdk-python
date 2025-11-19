"""Integration tests for BedrockModelInvoke."""

import pytest

from strands import Agent, tool
from strands.models.bedrock_invoke import BedrockModelInvoke


@tool
def string_length(string_to_measure: str) -> str:
    """Returns the length of the string passed in."""
    return str(len(string_to_measure))


@pytest.mark.skip(reason="Requires AWS credentials and imported model access")
def test_bedrock_invoke_with_imported_model():
    """Test BedrockModelInvoke with an imported model."""
    # Replace with actual imported model ARN
    imported_model_arn = "arn:aws:bedrock:us-east-1:123456789012:imported-model/tp8npa6a91gu"

    model = BedrockModelInvoke(model_id=imported_model_arn)
    agent = Agent(model, tools=[string_length])

    result = agent("Generate a random string, then tell me its length")

    # Verify the result has content
    assert result.message["content"]
    assert result.stop_reason == "end_turn"


@pytest.mark.skip(reason="Requires AWS credentials and model access")
def test_bedrock_invoke_basic_text_generation():
    """Test basic text generation without tools."""
    model = BedrockModelInvoke(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=100, temperature=0.7)

    agent = Agent(model)
    result = agent("Hello! Tell me a short joke.")

    # Verify the result has content
    assert result.message["content"]
    assert result.stop_reason == "end_turn"


@pytest.mark.skip(reason="Requires AWS credentials and model access")
def test_bedrock_invoke_with_system_prompt():
    """Test BedrockModelInvoke with system prompt."""
    model = BedrockModelInvoke(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=50)

    agent = Agent(model, system_prompt="You are a helpful assistant that responds in exactly 5 words.")
    result = agent("What is the weather like?")

    # Verify the result has content
    assert result.message["content"]
    assert result.stop_reason == "end_turn"


def test_bedrock_invoke_configuration():
    """Test BedrockModelInvoke configuration without making API calls."""
    model = BedrockModelInvoke(model_id="test-model", max_tokens=1000, temperature=0.5, streaming=False)

    config = model.get_config()
    assert config["model_id"] == "test-model"
    assert config["max_tokens"] == 1000
    assert config["temperature"] == 0.5
    assert config["streaming"] is False

    # Test config updates
    model.update_config(temperature=0.8, top_p=0.9)
    updated_config = model.get_config()
    assert updated_config["temperature"] == 0.8
    assert updated_config["top_p"] == 0.9
