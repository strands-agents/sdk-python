"""Live integration tests for the auxiliary model call hook pair and per-source metrics.

Covers both auxiliary streaming shapes against real Bedrock models: summarization
(``model.stream`` via ``generate_summary``) and routing classification
(``model.structured_output`` via ``ClassifierStrategy``).
"""

import pytest

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.hooks import AfterAuxiliaryModelCallEvent, BeforeAuxiliaryModelCallEvent
from strands.models import BedrockModel, ClassifierStrategy, ModelRouter, RoutingCandidate
from strands.types.exceptions import ModelThrottledException
from tests_integ.conftest import retry_on_flaky

_HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def aux_events():
    return []


@pytest.fixture
def aux_hooks(aux_events):
    return [
        (BeforeAuxiliaryModelCallEvent, aux_events.append),
        (AfterAuxiliaryModelCallEvent, aux_events.append),
    ]


def _register(agent, aux_hooks):
    for event_type, callback in aux_hooks:
        agent.hooks.add_callback(event_type, callback)


@retry_on_flaky(
    "Bedrock capacity may be transiently unavailable",
    max_attempts=2,
    retry_on=[ModelThrottledException],
)
def test_summarization_fires_hooks_and_records_usage(aux_events, aux_hooks):
    """Summarization (the ``model.stream`` path) fires the hook pair and books usage."""
    agent = Agent(
        model=BedrockModel(model_id=_HAIKU_MODEL_ID, max_tokens=512, streaming=False),
        conversation_manager=SummarizingConversationManager(summary_ratio=0.5, preserve_recent_messages=1),
        load_tools_from_directory=False,
        callback_handler=None,
    )
    _register(agent, aux_hooks)
    agent.messages = [
        {"role": "user", "content": [{"text": "My favorite color is teal and I live in Lisbon."}]},
        {"role": "assistant", "content": [{"text": "Noted: teal, Lisbon."}]},
        {"role": "user", "content": [{"text": "I also have a dog named Pixel."}]},
        {"role": "assistant", "content": [{"text": "Got it: a dog named Pixel."}]},
    ]

    agent.conversation_manager.reduce_context(agent)

    before, after = aux_events
    assert isinstance(before, BeforeAuxiliaryModelCallEvent)
    assert before.source == "summarization"
    assert isinstance(after, AfterAuxiliaryModelCallEvent)
    assert after.exception is None
    assert after.stop_response is not None
    assert after.stop_response.usage["totalTokens"] > 0

    summarization_usage = agent.event_loop_metrics.accumulated_usage_by_source["summarization"]
    assert summarization_usage["totalTokens"] == after.stop_response.usage["totalTokens"]


@retry_on_flaky(
    "Bedrock capacity may be transiently unavailable",
    max_attempts=2,
    retry_on=[ModelThrottledException],
)
def test_routing_classifier_fires_hooks_and_records_usage(aux_events, aux_hooks):
    """Routing classification (the ``model.structured_output`` path) fires the hook pair and books usage."""
    candidate_model = BedrockModel(model_id=_HAIKU_MODEL_ID, max_tokens=256, streaming=False)
    math_model = BedrockModel(model_id=_HAIKU_MODEL_ID, max_tokens=256, streaming=False)
    router = ModelRouter(
        models=[
            RoutingCandidate(candidate_model, name="general model"),
            RoutingCandidate(
                math_model,
                name="math model",
                description="Best suited to arithmetic questions.",
            ),
        ],
        strategy=ClassifierStrategy(
            BedrockModel(model_id=_HAIKU_MODEL_ID, max_tokens=64, streaming=False, temperature=0)
        ),
    )
    agent = Agent(
        model=router,
        system_prompt="Answer in one short sentence.",
        load_tools_from_directory=False,
        callback_handler=None,
    )
    _register(agent, aux_hooks)

    agent("What is 2 + 2?")

    routing_events = [event for event in aux_events if event.source == "routing"]
    before, after = routing_events
    assert isinstance(before, BeforeAuxiliaryModelCallEvent)
    assert isinstance(after, AfterAuxiliaryModelCallEvent)
    assert after.exception is None

    by_source = agent.event_loop_metrics.accumulated_usage_by_source
    assert by_source["main"]["totalTokens"] > 0
    assert by_source["routing"]["totalTokens"] > 0
    # Auxiliary usage stays out of the per-invocation figures the main loop tracks.
    invocation_total = sum(inv.usage["totalTokens"] for inv in agent.event_loop_metrics.agent_invocations)
    assert invocation_total == by_source["main"]["totalTokens"]
