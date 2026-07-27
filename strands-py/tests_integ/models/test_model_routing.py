"""Integration tests for model routing over real Bedrock models.

Ordered fallback is the routing behavior that only a real model surfaces: a real model failure
triggers a real recovery, observable purely through the answer. Proactive selection (context-fit,
first-candidate) is deterministic and covered by unit tests, so it is not re-tested here.
"""

import pytest

from strands import Agent
from strands.models import BedrockModel
from strands.models.routing import ModelRouter
from tests_integ.models import providers

# Routing runs over Bedrock candidates; skip when Bedrock is unavailable.
pytestmark = providers.bedrock.mark

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def haiku():
    return BedrockModel(model_id=_HAIKU_MODEL_ID)


@pytest.fixture
def broken_model():
    """A model whose id does not exist, so the first call raises before any tokens stream."""
    return BedrockModel(model_id="bogus.nonexistent-model-v1:0")


def test_fallback_recovers_a_real_answer_from_a_failing_primary(broken_model, haiku):
    """The broken primary fails, ordered fallback advances to the healthy model, and the task completes."""
    agent = Agent(model=ModelRouter(models=[broken_model, haiku]), load_tools_from_directory=False)

    result = agent("What is the capital of France? Reply with just the city name.")

    assert "paris" in result.message["content"][0]["text"].lower()
