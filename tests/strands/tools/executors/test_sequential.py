from concurrent.futures import ThreadPoolExecutor

import pytest

from strands.tools.executors import SequentialToolExecutor


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
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
        {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]},
        {"toolUseId": "2", "status": "success", "content": [{"text": "75F"}]},
        {"toolUseId": "2", "status": "success", "content": [{"text": "75F"}]},
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[1], exp_events[2]]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_sequential_executor_execute_threaded(
    agent, tool_results, cycle_trace, cycle_span, invocation_state, tool_events, alist
):
    tool_uses = [{"name": "thread_tool", "toolUseId": "1", "input": {}}]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_thread_pool") as thread_pool:
        executor = SequentialToolExecutor(thread_pool)
        stream = executor._execute(agent, tool_uses, tool_results, cycle_trace, cycle_span, invocation_state)

        await alist(stream)

        thread_name = tool_events[0]["thread_name"]
        assert thread_name.startswith("test_thread_pool")
