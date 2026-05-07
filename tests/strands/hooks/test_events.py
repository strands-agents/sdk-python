"""Tests for agent and multi-agent execution lifecycle events."""

from unittest.mock import Mock

import pytest

from strands.hooks import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    AfterReduceContextEvent,
    BaseHookEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    BeforeReduceContextEvent,
    HookEvent,
    MultiAgentInitializedEvent,
)


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


def test_before_node_call_event(orchestrator):
    """Test BeforeNodeCallEvent creation."""
    node_id = "node_1"
    event = BeforeNodeCallEvent(source=orchestrator, node_id=node_id)

    assert event.source is orchestrator
    assert event.node_id == node_id
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_before_multi_agent_invocation_event(orchestrator):
    """Test BeforeMultiAgentInvocationEvent creation."""
    event = BeforeMultiAgentInvocationEvent(source=orchestrator)

    assert event.source is orchestrator
    assert event.invocation_state is None
    assert isinstance(event, BaseHookEvent)


def test_after_events_should_reverse_callbacks(orchestrator):
    """Test that After events have should_reverse_callbacks property set to True."""
    after_node_event = AfterNodeCallEvent(source=orchestrator, node_id="test")
    after_invocation_event = AfterMultiAgentInvocationEvent(source=orchestrator)

    assert after_node_event.should_reverse_callbacks is True
    assert after_invocation_event.should_reverse_callbacks is True


@pytest.fixture
def agent():
    """Mock agent for testing."""
    return Mock()


def test_before_reduce_context_event_defaults(agent):
    """BeforeReduceContextEvent has sensible defaults and inherits from HookEvent."""
    event = BeforeReduceContextEvent(agent=agent)

    assert event.agent is agent
    assert event.exception is None
    assert event.message_count == 0
    assert event.should_reverse_callbacks is False
    assert isinstance(event, HookEvent)


def test_before_reduce_context_event_with_fields(agent):
    """BeforeReduceContextEvent carries the trigger exception and message count."""
    exc = RuntimeError("overflow")
    event = BeforeReduceContextEvent(agent=agent, exception=exc, message_count=12)

    assert event.exception is exc
    assert event.message_count == 12


def test_after_reduce_context_event_defaults(agent):
    """AfterReduceContextEvent has sensible defaults and runs callbacks in reverse."""
    event = AfterReduceContextEvent(agent=agent)

    assert event.agent is agent
    assert event.exception is None
    assert event.messages_removed == 0
    assert event.message_count_before == 0
    assert event.message_count_after == 0
    assert event.should_reverse_callbacks is True
    assert isinstance(event, HookEvent)


def test_after_reduce_context_event_with_fields(agent):
    """AfterReduceContextEvent carries before/after counts and the original exception."""
    exc = RuntimeError("overflow")
    event = AfterReduceContextEvent(
        agent=agent,
        exception=exc,
        messages_removed=3,
        message_count_before=10,
        message_count_after=7,
    )

    assert event.exception is exc
    assert event.messages_removed == 3
    assert event.message_count_before == 10
    assert event.message_count_after == 7