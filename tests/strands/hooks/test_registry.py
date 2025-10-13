import unittest.mock

import pytest

from strands.agent.interrupt import InterruptState
from strands.hooks import BeforeToolCallEvent, HookRegistry
from strands.interrupt import Interrupt


@pytest.fixture
def registry():
    return HookRegistry()


@pytest.fixture
def agent():
    instance = unittest.mock.Mock()
    instance.interrupt_state = InterruptState()
    return instance


def test_hook_registry_invoke_callbacks_interrupt(registry, agent):
    event = BeforeToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"toolUseId": "test_tool_id", "name": "test_tool_name", "input": {}},
        invocation_state={},
    )

    callback1 = unittest.mock.Mock(side_effect=lambda event: event.interrupt("test_name_1", "test reason 1"))
    callback2 = unittest.mock.Mock()
    callback3 = unittest.mock.Mock(side_effect=lambda event: event.interrupt("test_name_2", "test reason 2"))

    registry.add_callback(BeforeToolCallEvent, callback1)
    registry.add_callback(BeforeToolCallEvent, callback2)
    registry.add_callback(BeforeToolCallEvent, callback3)

    _, tru_interrupts = registry.invoke_callbacks(event)
    exp_interrupts = [
        Interrupt(
            id="v1:test_tool_id:da3551f3-154b-5978-827e-50ac387877ee",
            name="test_name_1",
            reason="test reason 1",
        ),
        Interrupt(
            id="v1:test_tool_id:0f5a8068-d1ba-5a48-bf67-c9d33786d8d4",
            name="test_name_2",
            reason="test reason 2",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    callback1.assert_called_once_with(event)
    callback2.assert_called_once_with(event)
    callback3.assert_called_once_with(event)


def test_hook_registry_invoke_callbacks_interrupt_name_clash(registry, agent):
    event = BeforeToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"toolUseId": "test_tool_id", "name": "test_tool_name", "input": {}},
        invocation_state={},
    )

    callback1 = unittest.mock.Mock(side_effect=lambda event: event.interrupt("test_name", "test reason 1"))
    callback2 = unittest.mock.Mock(side_effect=lambda event: event.interrupt("test_name", "test reason 2"))

    registry.add_callback(BeforeToolCallEvent, callback1)
    registry.add_callback(BeforeToolCallEvent, callback2)

    with pytest.raises(ValueError, match="interrupt_name=<test_name> | interrupt name used more than once"):
        registry.invoke_callbacks(event)
