"""Tests for Agent max_turns and max_token_budget execution limits."""

import threading
import unittest.mock

import pytest

from strands import Agent
from strands.event_loop._retry import ModelRetryStrategy
from strands.event_loop.event_loop import event_loop_cycle
from strands.interrupt import _InterruptState
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.types.exceptions import MaxTokenBudgetReachedException, MaxTurnsReachedException
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _text_response(text="done"):
    return {"role": "assistant", "content": [{"text": text}]}


def _tool_call_response(tool_name="noop", tool_use_id="t1"):
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": tool_name, "input": {}}}],
    }


# ---------------------------------------------------------------------------
# max_turns — validation
# ---------------------------------------------------------------------------


def test_max_turns_validation_rejects_zero():
    with pytest.raises(ValueError, match="max_turns must be a positive integer"):
        Agent(model=MockedModelProvider([_text_response()]), max_turns=0)


def test_max_turns_validation_rejects_negative():
    with pytest.raises(ValueError, match="max_turns must be a positive integer"):
        Agent(model=MockedModelProvider([_text_response()]), max_turns=-1)


# ---------------------------------------------------------------------------
# max_turns — runtime behavior
# ---------------------------------------------------------------------------


def test_max_turns_default_none_allows_execution():
    agent = Agent(model=MockedModelProvider([_text_response()]), load_tools_from_directory=False)
    result = agent("hello")
    assert result.stop_reason == "end_turn"


def test_max_turns_one_allows_single_cycle():
    agent = Agent(
        model=MockedModelProvider([_text_response()]),
        max_turns=1,
        load_tools_from_directory=False,
    )
    result = agent("hello")
    assert result.stop_reason == "end_turn"


def test_max_turns_stops_after_limit():
    """With max_turns=1, a tool-call response uses the first cycle; the follow-up
    cycle raises MaxTurnsReachedException before calling the model again."""
    from strands import tool as tool_decorator

    @tool_decorator
    def noop() -> str:
        """No-op tool."""
        return "ok"

    agent = Agent(
        model=MockedModelProvider(
            [
                _tool_call_response(tool_name="noop"),
                _text_response(),
            ]
        ),
        tools=[noop],
        max_turns=1,
        load_tools_from_directory=False,
    )

    with pytest.raises(MaxTurnsReachedException):
        agent("do something")


def test_max_turns_resets_between_invocations():
    """Counter should reset to 0 at the start of each invocation."""
    agent = Agent(
        model=MockedModelProvider([_text_response(), _text_response()]),
        max_turns=1,
        load_tools_from_directory=False,
    )
    # First invocation succeeds
    result = agent("first")
    assert result.stop_reason == "end_turn"
    # Second invocation also succeeds (counter was reset)
    result = agent("second")
    assert result.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# max_token_budget — validation
# ---------------------------------------------------------------------------


def test_max_token_budget_validation_rejects_zero():
    with pytest.raises(ValueError, match="max_token_budget must be a positive integer"):
        Agent(model=MockedModelProvider([_text_response()]), max_token_budget=0)


def test_max_token_budget_validation_rejects_negative():
    with pytest.raises(ValueError, match="max_token_budget must be a positive integer"):
        Agent(model=MockedModelProvider([_text_response()]), max_token_budget=-1)


# ---------------------------------------------------------------------------
# max_token_budget — runtime behavior
# ---------------------------------------------------------------------------


def test_max_token_budget_default_none_allows_execution():
    agent = Agent(model=MockedModelProvider([_text_response()]), load_tools_from_directory=False)
    result = agent("hello")
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_max_token_budget_stops_when_counter_already_at_limit():
    """Test the guard at the event_loop_cycle level directly, bypassing stream_async's reset.

    stream_async resets _invocation_token_count to 0 at the start of every invocation,
    so the only way to test the guard firing is to pre-set the counter on a mock agent
    and call event_loop_cycle directly — the same pattern used in test_event_loop.py."""
    mock_agent = unittest.mock.MagicMock()
    mock_agent.__class__ = Agent
    mock_agent.messages = []
    mock_agent.tool_registry = ToolRegistry()
    mock_agent.event_loop_metrics = EventLoopMetrics()
    mock_agent.event_loop_metrics.reset_usage_metrics()
    mock_agent.hooks.invoke_callbacks_async = unittest.mock.AsyncMock()
    mock_agent._interrupt_state = _InterruptState()
    mock_agent._cancel_signal = threading.Event()
    mock_agent._model_state = {}
    mock_agent.trace_attributes = {}
    mock_agent._retry_strategy = ModelRetryStrategy()
    mock_agent.max_turns = None
    mock_agent.max_token_budget = 500
    mock_agent._invocation_turn_count = 0
    mock_agent._invocation_token_count = 500  # already at limit

    with pytest.raises(MaxTokenBudgetReachedException):
        async for _ in event_loop_cycle(agent=mock_agent, invocation_state={}):
            pass


def test_max_token_budget_first_cycle_runs_when_counter_is_zero():
    """With counter at 0 and budget >= 1, the first cycle must always execute."""
    agent = Agent(
        model=MockedModelProvider([_text_response()]),
        max_token_budget=1,
        load_tools_from_directory=False,
    )
    # counter starts at 0; 0 >= 1 is False so first cycle runs
    result = agent("hello")
    assert result.stop_reason == "end_turn"


def test_max_token_budget_resets_between_invocations():
    """Token counter resets to 0 at the start of every invocation via stream_async."""
    agent = Agent(
        model=MockedModelProvider([_text_response(), _text_response()]),
        max_token_budget=500,
        load_tools_from_directory=False,
    )
    # Both calls should succeed because stream_async resets the counter to 0 each time
    result = agent("first")
    assert result.stop_reason == "end_turn"
    assert agent._invocation_token_count == 0  # MockedModelProvider emits no usage metadata

    result = agent("second")
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_max_token_budget_accumulates_from_message_metadata():
    """Token accumulation reads totalTokens from the message metadata after each model call."""
    mock_agent = unittest.mock.MagicMock()
    mock_agent.__class__ = Agent
    mock_agent.messages = []
    mock_agent.tool_registry = ToolRegistry()
    mock_agent.event_loop_metrics = EventLoopMetrics()
    mock_agent.event_loop_metrics.reset_usage_metrics()
    mock_agent.hooks.invoke_callbacks_async = unittest.mock.AsyncMock()
    mock_agent._interrupt_state = _InterruptState()
    mock_agent._cancel_signal = threading.Event()
    mock_agent._model_state = {}
    mock_agent.trace_attributes = {}
    mock_agent._retry_strategy = ModelRetryStrategy()
    mock_agent.max_turns = None
    mock_agent.max_token_budget = None
    mock_agent._invocation_turn_count = 0
    mock_agent._invocation_token_count = 0

    async def _stream_with_metadata(*args, **kwargs):
        yield {"contentBlockStart": {"start": {"text": ""}}}
        yield {"contentBlockDelta": {"delta": {"text": "hello"}}}
        yield {"contentBlockStop": {}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 150},
                "metrics": {"latencyMs": 100},
            }
        }

    mock_agent.model.stream = _stream_with_metadata

    events = []
    async for event in event_loop_cycle(agent=mock_agent, invocation_state={}):
        events.append(event)

    assert mock_agent._invocation_token_count == 150
