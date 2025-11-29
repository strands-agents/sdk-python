import unittest.mock

import pytest
import pytest_asyncio

from strands import tool
from strands.experimental.bidi.agent.loop import _BidiAgentLoop
from strands.experimental.bidi.models import BidiModelTimeoutError
from strands.experimental.bidi.types.events import BidiConnectionRestartEvent, BidiTextInputEvent
from strands.hooks import HookRegistry
from strands.tools.executors import SequentialToolExecutor
from strands.tools.registry import ToolRegistry
from strands.types._events import ToolResultEvent, ToolResultMessageEvent, ToolUseStreamEvent


@pytest.fixture
def time_tool():
    @tool(name="time_tool")
    async def func():
        return "12:00"

    return func


@pytest.fixture
def agent(time_tool):
    mock = unittest.mock.Mock()
    mock.hooks = HookRegistry()
    mock.messages = []
    mock.model = unittest.mock.AsyncMock()
    mock.tool_executor = SequentialToolExecutor()
    mock.tool_registry = ToolRegistry()
    mock.tool_registry.process_tools([time_tool])
    
    return mock


@pytest_asyncio.fixture
async def loop(agent):
    return _BidiAgentLoop(agent)


@pytest.mark.asyncio
async def test_bidi_agent_loop_receive_restart_connection(loop, agent, agenerator):
    timeout_error = BidiModelTimeoutError("test timeout")
    text_event = BidiTextInputEvent(text="test after restart")

    agent.model.receive = unittest.mock.Mock(side_effect=[timeout_error, agenerator([text_event])])

    await loop.start()
    
    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 2:
            break
    
    exp_events = [
        BidiConnectionRestartEvent(timeout_error),
        text_event,
    ]
    assert tru_events == exp_events
    
    agent.model.stop.assert_called_once()
    assert agent.model.start.call_count == 2
    agent.model.start.assert_any_call(
        agent.system_prompt,
        agent.tool_registry.get_all_tool_specs(),
        agent.messages,
    )


@pytest.mark.asyncio
async def test_bidi_agent_loop_receive_tool_use(loop, agent, agenerator):
    
    tool_use = {"toolUseId": "t1", "name": "time_tool", "input": {}}
    tool_result = {"toolUseId": "t1", "status": "success", "content": [{"text": "12:00"}]}

    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")
    tool_result_event = ToolResultEvent(tool_result)

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))
    
    await loop.start()
    
    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    exp_events = [
        tool_use_event,
        tool_result_event,
        ToolResultMessageEvent({"role": "user", "content": [{"toolResult": tool_result}]}),
    ]
    assert tru_events == exp_events
    
    tru_messages = agent.messages
    exp_messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
        {"role": "user", "content": [{"toolResult": tool_result}]},
    ]
    assert tru_messages == exp_messages

    agent.model.send.assert_called_with(tool_result_event)
