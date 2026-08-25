"""Tests for multi-agent execution lifecycle events."""

from unittest.mock import Mock

import pytest

from strands.hooks import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BaseHookEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    HookEvent,
    ModelStreamChunkEvent,
    MultiAgentInitializedEvent,
)


@pytest.fixture
def orchestrator():
    """Mock orchestrator for testing."""
    return Mock()


@pytest.fixture
def mock_agent():
    """Mock agent for testing."""
    return Mock()


def test_model_stream_chunk_event_creation(mock_agent):
    """Test ModelStreamChunkEvent creation with agent and chunk."""
    chunk = {"contentBlockDelta": {"delta": {"text": "Hello"}}}
    event = ModelStreamChunkEvent(agent=mock_agent, chunk=chunk)

    assert event.agent is mock_agent
    assert event.chunk == chunk
    assert isinstance(event, HookEvent)


def test_model_stream_chunk_event_with_message_start(mock_agent):
    """Test ModelStreamChunkEvent with messageStart chunk."""
    chunk = {"messageStart": {"role": "assistant"}}
    event = ModelStreamChunkEvent(agent=mock_agent, chunk=chunk)

    assert event.agent is mock_agent
    assert event.chunk == chunk


def test_model_stream_chunk_event_with_metadata(mock_agent):
    """Test ModelStreamChunkEvent with metadata chunk."""
    chunk = {
        "metadata": {
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            "metrics": {"latencyMs": 100},
        }
    }
    event = ModelStreamChunkEvent(agent=mock_agent, chunk=chunk)

    assert event.agent is mock_agent
    assert event.chunk == chunk


def test_model_stream_chunk_event_with_tool_use(mock_agent):
    """Test ModelStreamChunkEvent with tool use chunk."""
    chunk = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "123", "name": "test_tool"}}}}
    event = ModelStreamChunkEvent(agent=mock_agent, chunk=chunk)

    assert event.agent is mock_agent
    assert event.chunk == chunk


def test_model_stream_chunk_event_should_not_reverse_callbacks(mock_agent):
    """Test that ModelStreamChunkEvent does not reverse callbacks."""
    chunk = {"contentBlockDelta": {"delta": {"text": "test"}}}
    event = ModelStreamChunkEvent(agent=mock_agent, chunk=chunk)

    assert event.should_reverse_callbacks is False


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
