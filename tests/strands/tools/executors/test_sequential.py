import pytest

from strands.tools.executors.sequential import Executor as SAToolExecutor


@pytest.fixture
def executor():
    return SAToolExecutor()


@pytest.mark.asyncio
async def test_sequential_executor_execute(executor, agent, tool_results, invocation_state, alist):
    tool_uses = [
        {"name": "weather_tool", "toolUseId": "1", "input": {}},
        {"name": "temperature_tool", "toolUseId": "2", "input": {}},
    ]
    stream = executor.execute(agent, tool_uses, tool_results, invocation_state)

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
