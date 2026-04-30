import unittest.mock

import pytest

from strands.interrupt import CascadedInterruptException, Interrupt, _InterruptState
from strands.types.tools import ToolContext


@pytest.fixture
def agent():
    instance = unittest.mock.Mock()
    instance._interrupt_state = _InterruptState()
    return instance


def test_tool_context_cascade_interrupts_stores_interrupts_and_raises(agent):
    context = ToolContext(
        tool_use={"toolUseId": "tool_1", "name": "sub_agent_tool", "input": {}},
        agent=agent,
        invocation_state={},
    )
    interrupts = [
        Interrupt(id="sub-1", name="approval_1", reason="first"),
        Interrupt(id="sub-2", name="approval_2", reason="second"),
    ]

    with pytest.raises(CascadedInterruptException) as exc_info:
        context.cascade_interrupts(interrupts)

    assert exc_info.value.interrupts == interrupts
    assert agent._interrupt_state.interrupts["sub-1"] == interrupts[0]
    assert agent._interrupt_state.interrupts["sub-2"] == interrupts[1]
    assert agent._interrupt_state.context["cascaded:tool_1"] == ["sub-1", "sub-2"]


def test_tool_context_get_cascaded_interrupt_responses(agent):
    interrupt_1 = Interrupt(id="sub-1", name="approval_1", reason="first", response="approved")
    interrupt_2 = Interrupt(id="sub-2", name="approval_2", reason="second", response={"allow": True})
    agent._interrupt_state.interrupts = {
        interrupt_1.id: interrupt_1,
        interrupt_2.id: interrupt_2,
    }
    agent._interrupt_state.context = {"cascaded:tool_1": [interrupt_1.id, interrupt_2.id]}

    context = ToolContext(
        tool_use={"toolUseId": "tool_1", "name": "sub_agent_tool", "input": {}},
        agent=agent,
        invocation_state={},
    )

    assert context.get_cascaded_interrupt_responses() == [
        {"interruptResponse": {"interruptId": "sub-1", "response": "approved"}},
        {"interruptResponse": {"interruptId": "sub-2", "response": {"allow": True}}},
    ]


def test_tool_context_get_cascaded_interrupt_responses_when_not_resuming(agent):
    context = ToolContext(
        tool_use={"toolUseId": "tool_1", "name": "sub_agent_tool", "input": {}},
        agent=agent,
        invocation_state={},
    )

    assert context.get_cascaded_interrupt_responses() is None
