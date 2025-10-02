import pytest

from strands.hooks import BeforeToolCallEvent, Interrupt
from strands.tools.executors import SequentialToolExecutor
from strands.types._events import ToolInterruptEvent, ToolResultEvent


@pytest.fixture
def executor():
    return SequentialToolExecutor()


@pytest.mark.asyncio
async def test_sequential_executor_execute(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_uses = [
        {"name": "weather_tool", "toolUseId": "1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "2", "input": {}},
    ]
    stream = executor._execute(agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
        ToolResultEvent({"toolUseId": "2", "status": "success", "content": [{"text": "75F"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[0].tool_result, exp_events[1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_sequential_executor_interrupt(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    interrupt = Interrupt(
        name="weather_tool",
        event_name="BeforeToolCallEvent",
        reasons=["test reason"],
        activated=True,
    )

    def interrupt_callback(event):
        event.interrupt("test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    tool_uses = [
        {"name": "weather_tool", "toolUseId": "1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "2", "input": {}},
    ]

    stream = executor._execute(agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolInterruptEvent(interrupt),
        ToolResultEvent(
            {
                "toolUseId": "1",
                "status": "error",
                "content": [
                    {
                        "json": {
                            "interrupt": {
                                "name": "weather_tool",
                                "event_name": "BeforeToolCallEvent",
                                "reasons": ["test reason"],
                            },
                        },
                    },
                ],
            }
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results
