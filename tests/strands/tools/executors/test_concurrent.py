import pytest

from strands.tools.executors import ConcurrentToolExecutor
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolUse


@pytest.fixture
def executor():
    return ConcurrentToolExecutor()


@pytest.mark.asyncio
async def test_concurrent_executor_execute(
    executor, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_uses: list[ToolUse] = [
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
