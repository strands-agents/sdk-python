import pytest

from strands.hooks import BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.tools.executors import ConcurrentToolExecutor
from strands.types._events import ToolInterruptEvent, ToolResultEvent


@pytest.fixture
def executor():
    return ConcurrentToolExecutor()


@pytest.mark.asyncio
async def test_concurrent_executor_execute(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_uses = [
        {"name": "weather_tool", "toolUseId": "1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "2", "input": {}},
    ]
    stream = executor._execute(agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = sorted(await alist(stream), key=lambda event: event.tool_use_id)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
        ToolResultEvent({"toolUseId": "2", "status": "success", "content": [{"text": "75F"}]}),
    ]
    assert tru_events == exp_events

    tru_results = sorted(tool_results, key=lambda result: result.get("toolUseId"))
    exp_results = [exp_events[0].tool_result, exp_events[1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_concurrent_executor_interrupt(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    interrupt = Interrupt(
        id="v1:before_tool_call:test_tool_id_1:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
    )

    def interrupt_callback(event):
        if event.tool_use["name"] == "weather_tool":
            event.interrupt("test_name", "test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    tool_uses = [
        {"name": "weather_tool", "toolUseId": "test_tool_id_1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "test_tool_id_2", "input": {}},
    ]

    stream = executor._execute(agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = sorted(await alist(stream), key=lambda event: event.tool_use_id)
    exp_events = [
        ToolInterruptEvent(tool_uses[0], [interrupt]),
        ToolResultEvent({"toolUseId": "test_tool_id_2", "status": "success", "content": [{"text": "75F"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[1].tool_result]
    assert tru_results == exp_results
