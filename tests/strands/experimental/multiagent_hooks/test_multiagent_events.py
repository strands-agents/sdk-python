"""Tests for multi-agent execution lifecycle events."""

from unittest.mock import Mock

import pytest

from strands.experimental.multiagent_hooks.multiagent_events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeInvocationEvent,
    MultiAgentInitializationEvent,
)
from strands.hooks.registry import BaseHookEvent


@pytest.fixture
def orchestrator():
    """Mock orchestrator for testing."""
    return Mock()


def test_multi_agent_initialization_event_with_orchestrator_only(orchestrator):
    """Test MultiAgentInitializationEvent creation with orchestrator only."""
    event = MultiAgentInitializationEvent(orchestrator=orchestrator)

    assert event.orchestrator is orchestrator
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_multi_agent_initialization_event_with_invocation_state(orchestrator):
    """Test MultiAgentInitializationEvent creation with invocation state."""
    invocation_state = {"key": "value"}
    event = MultiAgentInitializationEvent(orchestrator=orchestrator, invocation_state=invocation_state)

    assert event.orchestrator is orchestrator
    assert event.invocation_state == invocation_state


def test_after_node_invocation_event_with_required_fields(orchestrator):
    """Test AfterNodeInvocationEvent creation with required fields."""
    executed_node = "node_1"
    event = AfterNodeInvocationEvent(orchestrator=orchestrator, executed_node=executed_node)

    assert event.orchestrator is orchestrator
    assert event.executed_node == executed_node
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_after_node_invocation_event_with_invocation_state(orchestrator):
    """Test AfterNodeInvocationEvent creation with invocation state."""
    executed_node = "node_2"
    invocation_state = {"result": "success"}
    event = AfterNodeInvocationEvent(
        orchestrator=orchestrator, executed_node=executed_node, invocation_state=invocation_state
    )

    assert event.orchestrator is orchestrator
    assert event.executed_node == executed_node
    assert event.invocation_state == invocation_state


def test_after_multi_agent_invocation_event_with_orchestrator_only(orchestrator):
    """Test AfterMultiAgentInvocationEvent creation with orchestrator only."""
    event = AfterMultiAgentInvocationEvent(orchestrator=orchestrator)

    assert event.orchestrator is orchestrator
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_after_multi_agent_invocation_event_with_invocation_state(orchestrator):
    """Test AfterMultiAgentInvocationEvent creation with invocation state."""
    invocation_state = {"final_state": "completed"}
    event = AfterMultiAgentInvocationEvent(orchestrator=orchestrator, invocation_state=invocation_state)

    assert event.orchestrator is orchestrator
    assert event.invocation_state == invocation_state
