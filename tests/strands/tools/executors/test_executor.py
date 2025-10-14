import unittest.mock
from unittest.mock import MagicMock

import pytest

import strands
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.telemetry.metrics import Trace
from strands.tools.executors._executor import ToolExecutor
from strands.types._events import ToolCancelEvent, ToolInterruptEvent, ToolResultEvent, ToolStreamEvent
from strands.types.tools import ToolUse


@pytest.fixture
def executor_cls():
    class ClsExecutor(ToolExecutor):
        def _execute(self, _agent, _tool_uses, _tool_results, _invocation_state):
            raise NotImplementedError

    return ClsExecutor


@pytest.fixture
def executor(executor_cls):
    return executor_cls()


@pytest.fixture
def tracer():
    with unittest.mock.patch.object(strands.tools.executors._executor, "get_tracer") as mock_get_tracer:
        yield mock_get_tracer.return_value


@pytest.mark.asyncio
async def test_executor_stream_yields_result(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_events = hook_events
    exp_hook_events = [
        BeforeToolCallEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
        ),
        AfterToolCallEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
            result=exp_results[0],
        ),
    ]
    assert tru_hook_events == exp_hook_events


@pytest.mark.asyncio
async def test_executor_stream_wraps_results(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    weather_tool.stream.return_value = agenerator(
        ["value 1", {"nested": True}, {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}]
    )

    tru_events = await alist(stream)
    exp_events = [
        ToolStreamEvent(tool_use, "value 1"),
        ToolStreamEvent(tool_use, {"nested": True}),
        ToolStreamEvent(tool_use, {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_passes_through_typed_events(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    event_1 = ToolStreamEvent(tool_use, "value 1")
    event_2 = ToolStreamEvent(tool_use, {"nested": True})
    event_3 = ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]})
    weather_tool.stream.return_value = agenerator(
        [
            event_1,
            event_2,
            event_3,
        ]
    )

    tru_events = await alist(stream)
    assert tru_events[0] is event_1
    assert tru_events[1] is event_2

    # ToolResults are not passed through directly, they're unwrapped then wraped again
    assert tru_events[2] == event_3


@pytest.mark.asyncio
async def test_executor_stream_wraps_stream_events_if_no_result(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    last_event = ToolStreamEvent(tool_use, "value 1")
    # Only ToolResultEvent can be the last value; all others are wrapped in ToolResultEvent
    weather_tool.stream.return_value = agenerator(
        [
            last_event,
        ]
    )

    tru_events = await alist(stream)
    exp_events = [last_event, ToolResultEvent(last_event)]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_yields_tool_error(
    executor, agent, tool_results, invocation_state, hook_events, exception_tool, alist
):
    tool_use = {"name": "exception_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [ToolResultEvent({"toolUseId": "1", "status": "error", "content": [{"text": "Error: Tool error"}]})]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolCallEvent(
        agent=agent,
        selected_tool=exception_tool,
        tool_use=tool_use,
        invocation_state=invocation_state,
        result=exp_results[0],
        exception=unittest.mock.ANY,
    )
    assert tru_hook_after_event == exp_hook_after_event


@pytest.mark.asyncio
async def test_executor_stream_yields_unknown_tool(executor, agent, tool_results, invocation_state, hook_events, alist):
    tool_use = {"name": "unknown_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "error", "content": [{"text": "Unknown tool: unknown_tool"}]})
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use=tool_use,
        invocation_state=invocation_state,
        result=exp_results[0],
    )
    assert tru_hook_after_event == exp_hook_after_event


@pytest.mark.asyncio
async def test_executor_stream_with_trace(
    executor, tracer, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream_with_trace(agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tracer.start_tool_call_span.assert_called_once_with(tool_use, cycle_span)
    tracer.end_tool_call_span.assert_called_once_with(
        tracer.start_tool_call_span.return_value,
        {"content": [{"text": "sunny"}], "status": "success", "toolUseId": "1"},
    )

    cycle_trace.add_child.assert_called_once()
    assert isinstance(cycle_trace.add_child.call_args[0][0], Trace)


@pytest.mark.parametrize(
    ("cancel_tool", "cancel_message"),
    [(True, "tool cancelled by user"), ("user cancel message", "user cancel message")],
)
@pytest.mark.asyncio
async def test_executor_stream_cancel(
    cancel_tool, cancel_message, executor, agent, tool_results, invocation_state, alist
):
    def cancel_callback(event):
        event.cancel_tool = cancel_tool
        return event

    agent.hooks.add_callback(BeforeToolCallEvent, cancel_callback)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolCancelEvent(tool_use, cancel_message),
        ToolResultEvent(
            {
                "toolUseId": "1",
                "status": "error",
                "content": [{"text": cancel_message}],
            },
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_interrupt(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "weather_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
    )

    def interrupt_callback(event):
        event.interrupt("test_name", reason="test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [ToolInterruptEvent(tool_use, [interrupt])]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = []
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_interrupt_resume(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "weather_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
        response="test response",
    )
    agent._interrupt_state[interrupt.id] = interrupt

    interrupt_response = {}

    def interrupt_callback(event):
        interrupt_response["response"] = event.interrupt("test_name", reason="test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent(
            {
                "toolUseId": "test_tool_id",
                "status": "success",
                "content": [{"text": "sunny"}],
            },
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_response = interrupt_response["response"]
    exp_response = "test response"
    assert tru_response == exp_response
