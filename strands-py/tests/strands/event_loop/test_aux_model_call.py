"""Tests for the shared auxiliary model call instrumentation helper."""

import pytest

from strands import Agent
from strands.event_loop._aux_model_call import instrument_aux_model_call
from strands.hooks import AfterAuxModelCallEvent, BeforeAuxModelCallEvent
from strands.types.content import Message
from strands.types.event_loop import Metrics, Usage
from strands.types.exceptions import AuxModelCallCancelledException
from tests.fixtures.mocked_model_provider import MockedModelProvider

STOP_MESSAGE: Message = {"role": "assistant", "content": [{"text": "response"}]}
STOP_USAGE = Usage(inputTokens=10, outputTokens=5, totalTokens=15)
STOP_METRICS = Metrics(latencyMs=7)


async def _stream_with_stop():
    yield {"contentBlockDelta": {"delta": {"text": "response"}}}
    yield {"stop": ("end_turn", STOP_MESSAGE, STOP_USAGE, STOP_METRICS)}


async def _stream_without_stop():
    yield {"output": "structured"}


async def _stream_that_raises(error):
    yield {"contentBlockDelta": {"delta": {"text": "partial"}}}
    raise error


@pytest.fixture
def agent():
    return Agent(model=MockedModelProvider([]))


@pytest.fixture
def hook_events(agent):
    received = []
    agent.hooks.add_callback(BeforeAuxModelCallEvent, received.append)
    agent.hooks.add_callback(AfterAuxModelCallEvent, received.append)
    return received


@pytest.mark.asyncio
async def test_fires_hook_pair_and_records_usage(agent, hook_events):
    messages = [{"role": "user", "content": [{"text": "summarize"}]}]
    invocation_state = {"key": "value"}

    events = [
        event
        async for event in instrument_aux_model_call(
            _stream_with_stop(),
            source="summarization",
            agent=agent,
            messages=messages,
            invocation_state=invocation_state,
        )
    ]

    exp_events = [
        {"contentBlockDelta": {"delta": {"text": "response"}}},
        {"stop": ("end_turn", STOP_MESSAGE, STOP_USAGE, STOP_METRICS)},
    ]
    assert events == exp_events

    before_event, after_event = hook_events
    assert isinstance(before_event, BeforeAuxModelCallEvent)
    assert before_event.source == "summarization"
    assert before_event.messages == messages
    assert before_event.invocation_state == invocation_state

    assert isinstance(after_event, AfterAuxModelCallEvent)
    assert after_event.source == "summarization"
    assert after_event.exception is None
    assert after_event.stop_response == AfterAuxModelCallEvent.ModelStopResponse(
        message=STOP_MESSAGE, stop_reason="end_turn", usage=STOP_USAGE
    )

    assert agent.event_loop_metrics.accumulated_usage == STOP_USAGE
    assert agent.event_loop_metrics.accumulated_usage_by_source == {"summarization": STOP_USAGE}


@pytest.mark.asyncio
async def test_agent_none_passes_events_through_uninstrumented():
    events = [
        event async for event in instrument_aux_model_call(_stream_with_stop(), source="summarization", agent=None)
    ]

    assert len(events) == 2


@pytest.mark.asyncio
async def test_cancel_raises_and_skips_after_event(agent, hook_events):
    def cancel(event: BeforeAuxModelCallEvent) -> None:
        event.cancel = True

    agent.hooks.add_callback(BeforeAuxModelCallEvent, cancel)

    with pytest.raises(AuxModelCallCancelledException, match="auxiliary model call cancelled by hook"):
        async for _ in instrument_aux_model_call(_stream_with_stop(), source="summarization", agent=agent):
            pass

    assert [type(event) for event in hook_events] == [BeforeAuxModelCallEvent]
    assert agent.event_loop_metrics.accumulated_usage_by_source == {}


@pytest.mark.asyncio
async def test_cancel_with_message_uses_it(agent):
    def cancel(event: BeforeAuxModelCallEvent) -> None:
        event.cancel = "blocked by guardrail"

    agent.hooks.add_callback(BeforeAuxModelCallEvent, cancel)

    with pytest.raises(AuxModelCallCancelledException, match="blocked by guardrail"):
        async for _ in instrument_aux_model_call(_stream_with_stop(), source="summarization", agent=agent):
            pass


@pytest.mark.asyncio
async def test_stream_error_fires_after_event_with_exception(agent, hook_events):
    error = RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        async for _ in instrument_aux_model_call(_stream_that_raises(error), source="summarization", agent=agent):
            pass

    before_event, after_event = hook_events
    assert isinstance(before_event, BeforeAuxModelCallEvent)
    assert isinstance(after_event, AfterAuxModelCallEvent)
    assert after_event.exception is error
    assert after_event.stop_response is None
    assert agent.event_loop_metrics.accumulated_usage_by_source == {}


@pytest.mark.asyncio
async def test_stream_without_stop_event_fires_hooks_without_usage(agent, hook_events):
    events = [
        event
        async for event in instrument_aux_model_call(_stream_without_stop(), source="routing_classifier", agent=agent)
    ]

    assert events == [{"output": "structured"}]

    before_event, after_event = hook_events
    assert before_event.source == "routing_classifier"
    assert after_event.stop_response is None
    assert after_event.exception is None
    assert agent.event_loop_metrics.accumulated_usage_by_source == {}


def test_before_event_cancel_is_writable_and_source_is_not(agent):
    event = BeforeAuxModelCallEvent(agent=agent, source="summarization")

    event.cancel = True
    assert event.cancel is True

    with pytest.raises(AttributeError, match="Property source is not writable"):
        event.source = "other"


def test_after_event_is_not_writable(agent):
    event = AfterAuxModelCallEvent(agent=agent, source="summarization")

    with pytest.raises(AttributeError, match="Property exception is not writable"):
        event.exception = RuntimeError("nope")
