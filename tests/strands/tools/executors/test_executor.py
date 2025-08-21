import unittest.mock

import pytest

import strands
from strands.experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
from strands.telemetry.metrics import Trace
from strands.tools.executors._executor import ToolExecutor


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
    tool_use = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1]]
    assert tru_results == exp_results

    tru_hook_events = hook_events
    exp_hook_events = [
        BeforeToolInvocationEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
        ),
        AfterToolInvocationEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
            result=exp_results[0],
        ),
    ]
    assert tru_hook_events == exp_hook_events


@pytest.mark.asyncio
async def test_executor_stream_yields_tool_error(
    executor, agent, tool_results, invocation_state, hook_events, exception_tool, alist
):
    tool_use = {"name": "exception_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [{"toolUseId": "1", "status": "error", "content": [{"text": "Error: Tool error"}]}]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1]]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolInvocationEvent(
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
    exp_events = [{"toolUseId": "1", "status": "error", "content": [{"text": "Unknown tool: unknown_tool"}]}]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1]]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolInvocationEvent(
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
    tool_use = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream_with_trace(agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1]]
    assert tru_results == exp_results

    tracer.start_tool_call_span.assert_called_once_with(tool_use, cycle_span)
    tracer.end_tool_call_span.assert_called_once_with(
        tracer.start_tool_call_span.return_value,
        {"content": [{"text": "sunny"}], "status": "success", "toolUseId": "1"},
    )

    cycle_trace.add_child.assert_called_once()
    assert isinstance(cycle_trace.add_child.call_args[0][0], Trace)
