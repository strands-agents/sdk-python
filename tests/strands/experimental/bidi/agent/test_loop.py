import unittest.mock

import pytest
import pytest_asyncio

from strands import tool
from strands.experimental.bidi import BidiAgent
from strands.experimental.bidi.agent.loop import _BidiAgentLoop
from strands.experimental.bidi.models import BidiModelTimeoutError
from strands.experimental.bidi.types.events import BidiConnectionRestartEvent, BidiTextInputEvent
from strands.types._events import ToolResultEvent, ToolResultMessageEvent, ToolUseStreamEvent


@pytest.fixture
def time_tool():
    @tool(name="time_tool")
    async def func():
        return "12:00"

    return func


@pytest.fixture
def agent(time_tool):
    return BidiAgent(model=unittest.mock.AsyncMock(), tools=[time_tool])


@pytest_asyncio.fixture
async def loop(agent):
    return agent._loop


@pytest.mark.asyncio
async def test_bidi_agent_loop_receive_restart_connection(loop, agent, agenerator):
    timeout_error = BidiModelTimeoutError("test timeout", test_restart_config=1)
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
    agent.model.start.assert_called_with(
        agent.system_prompt,
        agent.tool_registry.get_all_tool_specs(),
        agent.messages,
        test_restart_config=1,
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


@pytest.mark.asyncio
async def test_bidi_agent_loop_request_state_initialized_for_tools(loop, agent, agenerator):
    """Test that request_state is initialized before tool execution.

    This ensures tools that access request_state (like strands_tools.stop)
    work correctly even when invocation_state is not provided by the user.
    """

    @tool(name="check_request_state")
    async def check_request_state(request_state: dict) -> str:
        # Verify request_state exists and is writable
        request_state["test_key"] = "test_value"
        return f"keys: {list(request_state.keys())}"

    agent.tool_registry.register_tool(check_request_state)

    tool_use = {"toolUseId": "t2", "name": "check_request_state", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    # Start without providing invocation_state
    await loop.start()

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    # Verify tool executed successfully (request_state was available)
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"
    assert "test_key" in tool_result_event.tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_bidi_agent_loop_stop_event_loop_flag(loop, agent, agenerator):
    """Test that tools can set stop_event_loop flag to gracefully close connection."""

    @tool(name="stop_tool")
    async def stop_tool(request_state: dict) -> str:
        request_state["stop_event_loop"] = True
        return "stopping"

    agent.tool_registry.register_tool(stop_tool)

    tool_use = {"toolUseId": "t3", "name": "stop_tool", "input": {}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    await loop.start()

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)

    # Should receive: tool_use_event, tool_result_event, tool_result_message, connection_close
    assert len(tru_events) == 4

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"

    # Verify connection close event was emitted
    from strands.experimental.bidi.types.events import BidiConnectionCloseEvent

    connection_close_event = tru_events[3]
    assert isinstance(connection_close_event, BidiConnectionCloseEvent)
    assert connection_close_event["reason"] == "user_request"

    # Verify model.send was NOT called (tool result not sent to model)
    agent.model.send.assert_not_called()


@pytest.mark.asyncio
async def test_bidi_agent_loop_request_state_preserved_with_invocation_state(agent, agenerator):
    """Test that existing invocation_state is preserved when request_state is initialized."""

    @tool(name="check_invocation_state")
    async def check_invocation_state(custom_key: str) -> str:
        return f"custom_key: {custom_key}"

    agent.tool_registry.register_tool(check_invocation_state)

    tool_use = {"toolUseId": "t4", "name": "check_invocation_state", "input": {"custom_key": "from_state"}}
    tool_use_event = ToolUseStreamEvent(current_tool_use=tool_use, delta="")

    agent.model.receive = unittest.mock.Mock(return_value=agenerator([tool_use_event]))

    loop = agent._loop
    # Start with custom invocation_state but no request_state
    await loop.start(invocation_state={"custom_data": "preserved"})

    tru_events = []
    async for event in loop.receive():
        tru_events.append(event)
        if len(tru_events) >= 3:
            break

    # Verify tool executed successfully
    tool_result_event = tru_events[1]
    assert isinstance(tool_result_event, ToolResultEvent)
    assert tool_result_event.tool_result["status"] == "success"

    # Verify request_state was added without removing custom_data
    assert "request_state" in loop._invocation_state
    assert loop._invocation_state.get("custom_data") == "preserved"
