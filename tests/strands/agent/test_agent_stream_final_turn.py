"""Unit tests for the stream_final_turn_only parameter of Agent.stream_async.

Tests cover backward compatibility, single-turn and multi-turn invocations,
callback handler behavior, empty final turns, and non-text event passthrough.
"""

from unittest.mock import MagicMock, patch

import pytest

from strands import Agent
from strands.telemetry.metrics import EventLoopMetrics
from strands.types._events import (
    CitationStreamEvent,
    EventLoopStopEvent,
    InitEventLoopEvent,
    ModelStreamChunkEvent,
    ReasoningTextStreamEvent,
    StartEventLoopEvent,
    TextStreamEvent,
    ToolUseStreamEvent,
    TypedEvent,
)


@pytest.fixture
def mock_model():
    """Create a mock model for Agent construction."""
    model = MagicMock()
    model.stateful = False
    return model


@pytest.fixture
def callback_handler():
    """Create a mock callback handler."""
    return MagicMock()


@pytest.fixture
def agent(mock_model, callback_handler):
    """Create an Agent with mocked model and callback handler."""
    return Agent(
        model=mock_model,
        callback_handler=callback_handler,
        tools=[],
    )


def _make_text_event(text: str) -> TextStreamEvent:
    """Helper to create a TextStreamEvent."""
    return TextStreamEvent(text=text, delta={"text": text})


def _make_start_event_loop() -> StartEventLoopEvent:
    """Helper to create a StartEventLoopEvent."""
    return StartEventLoopEvent()


def _make_stop_event(stop_reason: str = "end_turn") -> EventLoopStopEvent:
    """Helper to create an EventLoopStopEvent."""
    return EventLoopStopEvent(
        stop_reason=stop_reason,
        message={"role": "assistant", "content": [{"text": "response"}]},
        metrics=EventLoopMetrics(),
        request_state={},
    )


def _make_init_event() -> InitEventLoopEvent:
    """Helper to create an InitEventLoopEvent."""
    return InitEventLoopEvent()


def _make_reasoning_event(text: str) -> ReasoningTextStreamEvent:
    """Helper to create a ReasoningTextStreamEvent."""
    return ReasoningTextStreamEvent(
        reasoning_text=text,
        delta={"reasoningContent": {"text": text}},
    )


def _make_citation_event() -> CitationStreamEvent:
    """Helper to create a CitationStreamEvent."""
    return CitationStreamEvent(
        delta={"citation": {"title": "source"}},
        citation={"title": "source"},
    )


def _make_tool_use_event() -> ToolUseStreamEvent:
    """Helper to create a ToolUseStreamEvent."""
    return ToolUseStreamEvent(
        delta={"toolUse": {"input": "{}"}},
        current_tool_use={"toolUseId": "t1", "name": "test_tool", "input": "{}"},
    )


def _make_model_stream_chunk_event() -> ModelStreamChunkEvent:
    """Helper to create a ModelStreamChunkEvent."""
    return ModelStreamChunkEvent(chunk={"contentBlockDelta": {"delta": {"text": "chunk"}}})


async def _mock_run_loop_from_events(events: list[TypedEvent]):
    """Create an async generator from a list of TypedEvent instances."""
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_stream_final_turn_only_false_yields_all_events(agent, callback_handler):
    """Test stream_final_turn_only=False yields all events unchanged (backward compatibility)."""
    text_event = _make_text_event("hello")
    start_event = _make_start_event_loop()
    stop_event = _make_stop_event("end_turn")
    init_event = _make_init_event()

    run_loop_events = [init_event, start_event, text_event, stop_event]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=False):
            yielded.append(event)

    # All callback events should be yielded (init, start, start_event_loop, text)
    # plus the AgentResultEvent at the end
    yielded_data_events = [e for e in yielded if "data" in e]
    assert len(yielded_data_events) == 1
    assert yielded_data_events[0]["data"] == "hello"

    # Callback handler should have been called with the text event
    text_calls = [c for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert len(text_calls) == 1
    assert text_calls[0].kwargs["data"] == "hello"


@pytest.mark.asyncio
async def test_single_turn_with_stream_final_turn_only_true(agent, callback_handler):
    """Test single-turn invocation with stream_final_turn_only=True yields all text events."""
    init_event = _make_init_event()
    start_event = _make_start_event_loop()
    text1 = _make_text_event("Hello ")
    text2 = _make_text_event("world!")
    stop_event = _make_stop_event("end_turn")

    run_loop_events = [init_event, start_event, text1, text2, stop_event]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=True):
            yielded.append(event)

    # Both text events should be yielded since this is the final (and only) turn
    yielded_data_events = [e for e in yielded if "data" in e]
    assert len(yielded_data_events) == 2
    assert yielded_data_events[0]["data"] == "Hello "
    assert yielded_data_events[1]["data"] == "world!"


@pytest.mark.asyncio
async def test_multi_turn_intermediate_text_suppressed_final_text_delivered(agent, callback_handler):
    """Test multi-turn: intermediate turn text suppressed, final turn text delivered."""
    init_event = _make_init_event()

    # Turn 1 (intermediate - tool_use)
    start1 = _make_start_event_loop()
    intermediate_text = _make_text_event("thinking...")
    stop1 = _make_stop_event("tool_use")

    # Turn 2 (final - end_turn)
    start2 = _make_start_event_loop()
    final_text1 = _make_text_event("Final ")
    final_text2 = _make_text_event("answer.")
    stop2 = _make_stop_event("end_turn")

    run_loop_events = [
        init_event,
        start1,
        intermediate_text,
        stop1,
        start2,
        final_text1,
        final_text2,
        stop2,
    ]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=True):
            yielded.append(event)

    # Only final turn text should appear
    yielded_data_events = [e for e in yielded if "data" in e]
    assert len(yielded_data_events) == 2
    assert yielded_data_events[0]["data"] == "Final "
    assert yielded_data_events[1]["data"] == "answer."

    # Intermediate text should NOT appear in callback calls
    all_callback_data = [c.kwargs.get("data") for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert "thinking..." not in all_callback_data
    assert "Final " in all_callback_data
    assert "answer." in all_callback_data


@pytest.mark.asyncio
async def test_callback_handler_receives_correct_events_false_mode(agent, callback_handler):
    """Test callback handler receives correct events when stream_final_turn_only=False."""
    init_event = _make_init_event()
    start_event = _make_start_event_loop()
    text_event = _make_text_event("hello")
    stop_event = _make_stop_event("end_turn")

    run_loop_events = [init_event, start_event, text_event, stop_event]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        async for _ in agent.stream_async("test", stream_final_turn_only=False):
            pass

    # Callback should have been called with the text event data
    text_calls = [c for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert len(text_calls) == 1
    assert text_calls[0].kwargs["data"] == "hello"


@pytest.mark.asyncio
async def test_callback_handler_receives_correct_events_true_mode(agent, callback_handler):
    """Test callback handler receives correct events when stream_final_turn_only=True."""
    init_event = _make_init_event()

    # Intermediate turn
    start1 = _make_start_event_loop()
    intermediate_text = _make_text_event("intermediate")
    stop1 = _make_stop_event("tool_use")

    # Final turn
    start2 = _make_start_event_loop()
    final_text = _make_text_event("final")
    stop2 = _make_stop_event("end_turn")

    run_loop_events = [init_event, start1, intermediate_text, stop1, start2, final_text, stop2]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        async for _ in agent.stream_async("test", stream_final_turn_only=True):
            pass

    # Only final text should reach callback
    all_callback_data = [c.kwargs.get("data") for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert all_callback_data == ["final"]


@pytest.mark.asyncio
async def test_empty_final_turn_no_text_events(agent, callback_handler):
    """Test empty final turn (no text events) produces no errors and yields zero text events."""
    init_event = _make_init_event()
    start_event = _make_start_event_loop()
    stop_event = _make_stop_event("end_turn")

    run_loop_events = [init_event, start_event, stop_event]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=True):
            yielded.append(event)

    # No text events should be yielded
    yielded_data_events = [e for e in yielded if "data" in e]
    assert len(yielded_data_events) == 0

    # No text callback calls
    text_calls = [c for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert len(text_calls) == 0


@pytest.mark.asyncio
async def test_non_text_events_pass_through_in_all_turns(agent, callback_handler):
    """Test non-text events pass through in all turns when stream_final_turn_only=True."""
    init_event = _make_init_event()

    # Intermediate turn with reasoning and tool use events
    start1 = _make_start_event_loop()
    reasoning_event = _make_reasoning_event("let me think")
    tool_use_event = _make_tool_use_event()
    intermediate_text = _make_text_event("intermediate text")
    model_chunk = _make_model_stream_chunk_event()
    stop1 = _make_stop_event("tool_use")

    # Final turn with citation event
    start2 = _make_start_event_loop()
    citation_event = _make_citation_event()
    final_text = _make_text_event("final text")
    stop2 = _make_stop_event("end_turn")

    run_loop_events = [
        init_event,
        start1,
        reasoning_event,
        tool_use_event,
        intermediate_text,
        model_chunk,
        stop1,
        start2,
        citation_event,
        final_text,
        stop2,
    ]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=True):
            yielded.append(event)

    # Reasoning event from intermediate turn should be present
    reasoning_events = [e for e in yielded if "reasoningText" in e]
    assert len(reasoning_events) == 1
    assert reasoning_events[0]["reasoningText"] == "let me think"

    # Tool use event from intermediate turn should be present
    tool_events = [e for e in yielded if e.get("type") == "tool_use_stream"]
    assert len(tool_events) == 1

    # Citation event from final turn should be present
    citation_events = [e for e in yielded if "citation" in e]
    assert len(citation_events) == 1

    # Model stream chunk events should be present (they have "event" key)
    chunk_events = [e for e in yielded if "event" in e and "contentBlockDelta" in e.get("event", {})]
    assert len(chunk_events) == 1

    # Intermediate text should NOT be present, final text should be
    data_events = [e for e in yielded if "data" in e]
    assert len(data_events) == 1
    assert data_events[0]["data"] == "final text"


@pytest.mark.asyncio
async def test_multiple_intermediate_turns_only_final_text_delivered(agent, callback_handler):
    """Test multiple intermediate turns: all intermediate text discarded, only final text delivered."""
    init_event = _make_init_event()

    # Turn 1 (intermediate)
    start1 = _make_start_event_loop()
    text1 = _make_text_event("turn 1 text")
    stop1 = _make_stop_event("tool_use")

    # Turn 2 (intermediate)
    start2 = _make_start_event_loop()
    text2 = _make_text_event("turn 2 text")
    stop2 = _make_stop_event("tool_use")

    # Turn 3 (final)
    start3 = _make_start_event_loop()
    text3 = _make_text_event("final answer")
    stop3 = _make_stop_event("end_turn")

    run_loop_events = [
        init_event,
        start1,
        text1,
        stop1,
        start2,
        text2,
        stop2,
        start3,
        text3,
        stop3,
    ]

    with patch.object(agent, "_run_loop", return_value=_mock_run_loop_from_events(run_loop_events)):
        yielded = []
        async for event in agent.stream_async("test", stream_final_turn_only=True):
            yielded.append(event)

    data_events = [e for e in yielded if "data" in e]
    assert len(data_events) == 1
    assert data_events[0]["data"] == "final answer"

    all_callback_data = [c.kwargs.get("data") for c in callback_handler.call_args_list if "data" in c.kwargs]
    assert all_callback_data == ["final answer"]
