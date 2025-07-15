import unittest.mock
from dataclasses import dataclass
from typing import List
from unittest.mock import MagicMock, Mock

import pytest

from strands.hooks import HookEvent, HookProvider, HookRegistry


@dataclass
class TestEvent(HookEvent):
    @property
    def should_reverse_callbacks(self) -> bool:
        return False


@dataclass
class TestAfterEvent(HookEvent):
    @property
    def should_reverse_callbacks(self) -> bool:
        return True


class TestHookProvider(HookProvider):
    """Test hook provider for testing hook registry."""

    def __init__(self):
        self.registered = False

    def register_hooks(self, registry: HookRegistry) -> None:
        self.registered = True


@pytest.fixture
def hook_registry():
    return HookRegistry()


@pytest.fixture
def test_event():
    return TestEvent(agent=Mock())


@pytest.fixture
def test_after_event():
    return TestAfterEvent(agent=Mock())


def test_hook_registry_init():
    """Test that HookRegistry initializes with an empty callbacks dictionary."""
    registry = HookRegistry()
    assert registry._registered_callbacks == {}


def test_add_callback(hook_registry, test_event):
    """Test that callbacks can be added to the registry."""
    callback = unittest.mock.Mock()
    hook_registry.add_callback(TestEvent, callback)

    assert TestEvent in hook_registry._registered_callbacks
    assert callback in hook_registry._registered_callbacks[TestEvent]


def test_add_multiple_callbacks_same_event(hook_registry, test_event):
    """Test that multiple callbacks can be added for the same event type."""
    callback1 = unittest.mock.Mock()
    callback2 = unittest.mock.Mock()

    hook_registry.add_callback(TestEvent, callback1)
    hook_registry.add_callback(TestEvent, callback2)

    assert len(hook_registry._registered_callbacks[TestEvent]) == 2
    assert callback1 in hook_registry._registered_callbacks[TestEvent]
    assert callback2 in hook_registry._registered_callbacks[TestEvent]


def test_add_hook(hook_registry):
    """Test that hooks can be added to the registry."""
    hook_provider = MagicMock()
    hook_registry.add_hook(hook_provider)

    assert hook_provider.register_hooks.call_count == 1


def test_get_callbacks_for_normal_event(hook_registry, test_event):
    """Test that get_callbacks_for returns callbacks in the correct order for normal events."""
    callback1 = unittest.mock.Mock()
    callback2 = unittest.mock.Mock()

    hook_registry.add_callback(TestEvent, callback1)
    hook_registry.add_callback(TestEvent, callback2)

    callbacks = list(hook_registry.get_callbacks_for(test_event))

    assert len(callbacks) == 2
    assert callbacks[0] == callback1
    assert callbacks[1] == callback2


def test_get_callbacks_for_after_event(hook_registry, test_after_event):
    """Test that get_callbacks_for returns callbacks in reverse order for after events."""
    callback1 = Mock()
    callback2 = Mock()

    hook_registry.add_callback(TestAfterEvent, callback1)
    hook_registry.add_callback(TestAfterEvent, callback2)

    callbacks = list(hook_registry.get_callbacks_for(test_after_event))

    assert len(callbacks) == 2
    assert callbacks[0] == callback2  # Reverse order
    assert callbacks[1] == callback1  # Reverse order


def test_invoke_callbacks(hook_registry, test_event):
    """Test that invoke_callbacks calls all registered callbacks for an event."""
    callback1 = Mock()
    callback2 = Mock()

    hook_registry.add_callback(TestEvent, callback1)
    hook_registry.add_callback(TestEvent, callback2)

    hook_registry.invoke_callbacks(test_event)

    callback1.assert_called_once_with(test_event)
    callback2.assert_called_once_with(test_event)


def test_invoke_callbacks_no_registered_callbacks(hook_registry, test_event):
    """Test that invoke_callbacks doesn't fail when there are no registered callbacks."""
    # No callbacks registered
    hook_registry.invoke_callbacks(test_event)
    # Test passes if no exception is raised


def test_invoke_callbacks_after_event(hook_registry, test_after_event):
    """Test that invoke_callbacks calls callbacks in reverse order for after events."""
    call_order: List[str] = []

    def callback1(_event):
        call_order.append("callback1")

    def callback2(_event):
        call_order.append("callback2")

    hook_registry.add_callback(TestAfterEvent, callback1)
    hook_registry.add_callback(TestAfterEvent, callback2)

    hook_registry.invoke_callbacks(test_after_event)

    assert call_order == ["callback2", "callback1"]  # Reverse order


def test_has_callbacks(hook_registry, test_event):
    """Test that has_callbacks returns correct boolean values."""
    # Empty registry should return False
    assert not hook_registry.has_callbacks()

    # Registry with callbacks should return True
    callback = Mock()
    hook_registry.add_callback(TestEvent, callback)
    assert hook_registry.has_callbacks()

    # Test with multiple event types
    hook_registry.add_callback(TestAfterEvent, Mock())
    assert hook_registry.has_callbacks()
