"""Tests for multi-agent execution lifecycle events."""

from unittest.mock import Mock

import pytest

from strands.experimental.hooks.multiagent.events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    MultiAgentInitializedEvent,
)
from strands.hooks.registry import BaseHookEvent


@pytest.fixture
def orchestrator():
    """Mock orchestrator for testing."""
    return Mock()


def test_multi_agent_initialization_event_with_orchestrator_only(orchestrator):
    """Test MultiAgentInitializedEvent creation with orchestrator only."""
    event = MultiAgentInitializedEvent(source=orchestrator)

    assert event.source is orchestrator
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_multi_agent_initialization_event_with_invocation_state(orchestrator):
    """Test MultiAgentInitializedEvent creation with invocation state."""
    invocation_state = {"key": "value"}
    event = MultiAgentInitializedEvent(source=orchestrator, invocation_state=invocation_state)

    assert event.source is orchestrator
    assert event.invocation_state == invocation_state


def test_after_node_invocation_event_with_required_fields(orchestrator):
    """Test AfterNodeCallEvent creation with required fields."""
    node_id = "node_1"
    event = AfterNodeCallEvent(source=orchestrator, node_id=node_id)

    assert event.source is orchestrator
    assert event.node_id == node_id
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_after_node_invocation_event_with_invocation_state(orchestrator):
    """Test AfterNodeCallEvent creation with invocation state."""
    node_id = "node_2"
    invocation_state = {"result": "success"}
    event = AfterNodeCallEvent(source=orchestrator, node_id=node_id, invocation_state=invocation_state)

    assert event.source is orchestrator
    assert event.node_id == node_id
    assert event.invocation_state == invocation_state


def test_after_multi_agent_invocation_event_with_orchestrator_only(orchestrator):
    """Test AfterMultiAgentInvocationEvent creation with orchestrator only."""
    event = AfterMultiAgentInvocationEvent(source=orchestrator)

    assert event.source is orchestrator
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_after_multi_agent_invocation_event_with_invocation_state(orchestrator):
    """Test AfterMultiAgentInvocationEvent creation with invocation state."""
    invocation_state = {"final_state": "completed"}
    event = AfterMultiAgentInvocationEvent(source=orchestrator, invocation_state=invocation_state)

    assert event.source is orchestrator
    assert event.invocation_state == invocation_state
