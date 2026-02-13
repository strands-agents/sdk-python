import unittest.mock

import pytest

from strands.hooks import AgentInitializedEvent, BeforeInvocationEvent, BeforeToolCallEvent, HookRegistry
from strands.interrupt import Interrupt, _InterruptState


@pytest.fixture
def registry():
    return HookRegistry()


@pytest.fixture
def agent():
    instance = unittest.mock.Mock()
    instance._interrupt_state = _InterruptState()
    return instance


def test_hook_registry_add_callback_agent_init_coroutine(registry):
    callback = unittest.mock.AsyncMock()

    with pytest.raises(ValueError, match=r"AgentInitializedEvent can only be registered with a synchronous callback"):
        registry.add_callback(AgentInitializedEvent, callback)


@pytest.mark.asyncio
async def test_hook_registry_invoke_callbacks_async_interrupt(registry, agent):
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

    _, tru_interrupts = await registry.invoke_callbacks_async(event)
    exp_interrupts = [
        Interrupt(
            id="v1:before_tool_call:test_tool_id:da3551f3-154b-5978-827e-50ac387877ee",
            name="test_name_1",
            reason="test reason 1",
        ),
        Interrupt(
            id="v1:before_tool_call:test_tool_id:0f5a8068-d1ba-5a48-bf67-c9d33786d8d4",
            name="test_name_2",
            reason="test reason 2",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    callback1.assert_called_once_with(event)
    callback2.assert_called_once_with(event)
    callback3.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_hook_registry_invoke_callbacks_async_interrupt_name_clash(registry, agent):
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
        await registry.invoke_callbacks_async(event)


def test_hook_registry_invoke_callbacks_coroutine(registry, agent):
    callback = unittest.mock.AsyncMock()
    registry.add_callback(BeforeInvocationEvent, callback)

    with pytest.raises(RuntimeError, match=r"use invoke_callbacks_async to invoke async callback"):
        registry.invoke_callbacks(BeforeInvocationEvent(agent=agent))


def test_hook_registry_add_callback_infers_event_type(registry):
    """Test that add_callback infers event type from callback type hint."""

    def typed_callback(event: BeforeInvocationEvent) -> None:
        pass

    # Register without explicit event_type - should infer from type hint
    registry.add_callback(typed_callback)

    # Verify callback was registered
    assert BeforeInvocationEvent in registry._registered_callbacks
    assert typed_callback in registry._registered_callbacks[BeforeInvocationEvent]


def test_hook_registry_add_callback_raises_error_no_type_hint(registry):
    """Test that add_callback raises error when type hint is missing."""

    def untyped_callback(event):
        pass

    with pytest.raises(ValueError, match="cannot infer event type"):
        registry.add_callback(untyped_callback)


def test_hook_registry_add_callback_raises_error_invalid_type_hint(registry):
    """Test that add_callback raises error when type hint is not a BaseHookEvent subclass."""

    def invalid_callback(event: str) -> None:
        pass

    with pytest.raises(ValueError, match="must be a subclass of BaseHookEvent"):
        registry.add_callback(invalid_callback)


def test_hook_registry_add_callback_raises_error_no_parameters(registry):
    """Test that add_callback raises error when callback has no parameters."""

    def no_param_callback() -> None:
        pass

    with pytest.raises(ValueError, match="callback has no parameters"):
        registry.add_callback(no_param_callback)
