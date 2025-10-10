import unittest.mock

import pytest

from strands.hooks import BeforeToolCallEvent, HookRegistry, Interrupt


@pytest.fixture
def registry():
    return HookRegistry()


def test_hook_registry_invoke_callbacks_interrupt(registry):
    interrupt = Interrupt(name="test", event_name="BeforeToolCallEvent", reasons=[])
    event = BeforeToolCallEvent(
        agent=unittest.mock.Mock(),
        selected_tool=None,
        tool_use={"toolUseId": "test", "name": "test_tool", "input": {}},
        invocation_state={},
        interrupt=interrupt,
    )

    callback1 = unittest.mock.Mock(side_effect=lambda event: event.interrupt("test reason"))
    callback2 = unittest.mock.Mock()

    registry.add_callback(BeforeToolCallEvent, callback1)
    registry.add_callback(BeforeToolCallEvent, callback2)

    registry.invoke_callbacks(event)

    callback1.assert_called_once_with(event)
    callback2.assert_called_once_with(event)
    assert interrupt.activated
