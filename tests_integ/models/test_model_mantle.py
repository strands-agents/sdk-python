"""Integration tests for OpenAI-compatible APIs on Bedrock Mantle.

Exercises the ``bedrock_mantle_config`` pathway on ``OpenAIModel`` (Chat Completions) and
``OpenAIResponsesModel`` (Responses API) against the live
``bedrock-mantle.<region>.api.aws/v1`` endpoint. Credentials come from the
ambient AWS credential chain; no explicit API key is passed by the user.
"""

import pytest

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.models.openai_responses import OpenAIResponsesModel

_REGION = "us-east-1"
_MODEL_ID = "openai.gpt-oss-120b"


@pytest.fixture
def bedrock_mantle_config():
    return {"region": _REGION}


@pytest.fixture
def chat_completions_model(bedrock_mantle_config):
    return OpenAIModel(model_id=_MODEL_ID, bedrock_mantle_config=bedrock_mantle_config)


@pytest.fixture
def model(bedrock_mantle_config):
    return OpenAIResponsesModel(model_id=_MODEL_ID, bedrock_mantle_config=bedrock_mantle_config)


@pytest.fixture
def stateful_model(bedrock_mantle_config):
    return OpenAIResponsesModel(model_id=_MODEL_ID, stateful=True, bedrock_mantle_config=bedrock_mantle_config)


def test_chat_completions_agent_invoke(chat_completions_model):
    """OpenAIModel (Chat Completions) reaches Mantle via bedrock_mantle_config."""
    agent = Agent(model=chat_completions_model, system_prompt="Reply in one short sentence.", callback_handler=None)
    result = agent("What is 2+2?")
    assert "4" in str(result)


def test_agent_invoke(model):
    agent = Agent(model=model, system_prompt="Reply in one short sentence.", callback_handler=None)
    result = agent("What is 2+2?")
    assert "4" in str(result)


def test_responses_server_side_conversation(stateful_model):
    agent = Agent(model=stateful_model, system_prompt="Reply in one short sentence.", callback_handler=None)

    agent("My name is Alice.")
    assert len(agent.messages) == 0

    result = agent("What is my name?")
    assert "alice" in str(result).lower()


def test_reasoning_content_multi_turn(bedrock_mantle_config):
    """Test that reasoning content from gpt-oss models doesn't break multi-turn conversations."""
    model = OpenAIResponsesModel(
        model_id=_MODEL_ID,
        bedrock_mantle_config=bedrock_mantle_config,
        params={"reasoning": {"effort": "low"}},
    )
    agent = Agent(model=model, system_prompt="Reply in one short sentence.", callback_handler=None)

    result1 = agent("What is 2+2?")
    assert "4" in str(result1)

    # Verify reasoning content was produced
    has_reasoning = any(
        "reasoningContent" in block for msg in agent.messages if msg["role"] == "assistant" for block in msg["content"]
    )
    assert has_reasoning

    # Second turn should not raise despite reasoningContent in message history
    agent("What about 3+3?")
