"""Tests for multi-agent session persistence hook implementation."""

from unittest.mock import Mock, call

import pytest

from strands.experimental.multiagent_hooks.multiagent_events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
)
from strands.experimental.multiagent_hooks.persistence_hooks import PersistentHook
from strands.hooks.registry import HookRegistry


@pytest.fixture
def session_manager():
    """Mock session manager."""
    return Mock()


@pytest.fixture
def orchestrator():
    """Mock orchestrator."""
    mock_orchestrator = Mock()
    mock_orchestrator.get_state_from_orchestrator.return_value = {"state": "test"}
    return mock_orchestrator


@pytest.fixture
def hook(session_manager):
    """PersistentHook instance."""
    return PersistentHook(session_manager)


def test_initialization(session_manager):
    """Test hook initialization."""
    hook = PersistentHook(session_manager)

    assert hook._session_manager is session_manager
    assert hasattr(hook, "_lock")


def test_register_hooks(hook):
    """Test hook registration with registry."""
    registry = Mock(spec=HookRegistry)

    hook.register_hooks(registry)

    expected_calls = [
        call(MultiAgentInitializationEvent, hook._on_initialization),
        call(BeforeMultiAgentInvocationEvent, hook._on_before_multiagent),
        call(BeforeNodeInvocationEvent, hook._on_before_node),
        call(AfterNodeInvocationEvent, hook._on_after_node),
        call(AfterMultiAgentInvocationEvent, hook._on_after_multiagent),
    ]
    registry.add_callback.assert_has_calls(expected_calls)


def test_on_initialization_persists_state(hook, orchestrator):
    """Test initialization event triggers persistence."""
    event = MultiAgentInitializationEvent(orchestrator=orchestrator)

    hook._on_initialization(event)

    orchestrator.get_state_from_orchestrator.assert_called_once()
    hook._session_manager.write_multi_agent_json.assert_called_once_with({"state": "test"})


def test_on_before_multiagent_does_nothing(hook, orchestrator):
    """Test before multiagent event does nothing."""
    event = BeforeMultiAgentInvocationEvent(orchestrator=orchestrator)

    hook._on_before_multiagent(event)

    orchestrator.get_state_from_orchestrator.assert_not_called()
    hook._session_manager.write_multi_agent_json.assert_not_called()


def test_on_before_node_does_nothing(hook, orchestrator):
    """Test before node event does nothing."""
    event = BeforeNodeInvocationEvent(orchestrator=orchestrator, next_node_to_execute="node_1")

    hook._on_before_node(event)

    orchestrator.get_state_from_orchestrator.assert_not_called()
    hook._session_manager.write_multi_agent_json.assert_not_called()


def test_on_after_node_persists_state(hook, orchestrator):
    """Test after node event triggers persistence."""
    event = AfterNodeInvocationEvent(orchestrator=orchestrator, executed_node="node_1")

    hook._on_after_node(event)

    orchestrator.get_state_from_orchestrator.assert_called_once()
    hook._session_manager.write_multi_agent_json.assert_called_once_with({"state": "test"})


def test_on_after_multiagent_persists_state(hook, orchestrator):
    """Test after multiagent event triggers persistence."""
    event = AfterMultiAgentInvocationEvent(orchestrator=orchestrator)

    hook._on_after_multiagent(event)

    orchestrator.get_state_from_orchestrator.assert_called_once()
    hook._session_manager.write_multi_agent_json.assert_called_once_with({"state": "test"})


def test_persist_thread_safety(hook, orchestrator):
    """Test that persistence operations are thread-safe."""
    hook._lock = Mock()
    hook._lock.__enter__ = Mock(return_value=hook._lock)
    hook._lock.__exit__ = Mock(return_value=None)

    hook._persist(orchestrator)

    hook._lock.__enter__.assert_called_once()
    hook._lock.__exit__.assert_called_once()
    orchestrator.get_state_from_orchestrator.assert_called_once()
    hook._session_manager.write_multi_agent_json.assert_called_once_with({"state": "test"})


def test_persist_gets_state_and_writes(hook, orchestrator):
    """Test persist method gets state and writes to session manager."""
    hook._persist(orchestrator)

    orchestrator.get_state_from_orchestrator.assert_called_once()
    hook._session_manager.write_multi_agent_json.assert_called_once_with({"state": "test"})
