import unittest.mock

import pytest
import pytest_asyncio

from strands.experimental.bidi.agent.loop import _BidiAgentLoop
from strands.experimental.bidi.models import BidiModelTimeoutError
from strands.experimental.bidi.types.events import BidiConnectionRestartEvent, BidiTextInputEvent


@pytest.fixture
def agent():
    mock = unittest.mock.Mock()
    mock.hooks = unittest.mock.AsyncMock()
    mock.model = unittest.mock.AsyncMock()
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
        agent.tool_registry.get_all_tool_specs.return_value,
        agent.messages,
    )
