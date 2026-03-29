"""Unit tests for Swarm AgentBase protocol support.

Tests that Swarm correctly handles AgentBase implementations that are not
concrete Agent instances, following the same patterns as Graph.
"""

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from strands.agent import Agent, AgentResult
from strands.agent.state import AgentState
from strands.hooks.registry import HookRegistry
from strands.interrupt import Interrupt, _InterruptState
from strands.multiagent.base import Status
from strands.multiagent.swarm import Swarm, SwarmNode


class MockAgentBase:
    """Mock implementation of AgentBase protocol for testing.

    Provides the minimal AgentBase interface without Agent-specific attributes
    like messages, state, tool_registry, etc.
    """

    def __init__(self, name: str, response_text: str = "AgentBase response"):
        """Initialize mock AgentBase.

        Args:
            name: Name of the agent.
            response_text: Text to return in responses.
        """
        self.name = name
        self._response_text = response_text
        self._call_count = 0

    def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Synchronous invocation."""
        return asyncio.run(self.invoke_async(prompt, **kwargs))

    async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Asynchronous invocation."""
        self._call_count += 1
        return AgentResult(
            message={"role": "assistant", "content": [{"text": self._response_text}]},
            stop_reason="end_turn",
            state={},
            metrics=Mock(
                accumulated_usage={"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                accumulated_metrics={"latencyMs": 50.0},
            ),
        )

    async def stream_async(self, prompt: Any = None, **kwargs: Any):
        """Stream agent execution asynchronously."""
        yield {"agent_start": True, "node": self.name}
        result = await self.invoke_async(prompt, **kwargs)
        yield {"result": result}


class NamelessAgentBase:
    """AgentBase without a name attribute."""

    async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Asynchronous invocation."""
        return AgentResult(
            message={"role": "assistant", "content": [{"text": "response"}]},
            stop_reason="end_turn",
            state={},
            metrics=None,
        )

    def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Synchronous invocation."""
        return asyncio.run(self.invoke_async(prompt, **kwargs))

    async def stream_async(self, prompt: Any = None, **kwargs: Any):
        """Stream agent execution asynchronously."""
        result = await self.invoke_async(prompt, **kwargs)
        yield {"result": result}


def create_mock_agent(name: str, response_text: str = "Agent response") -> Agent:
    """Create a mock Agent for comparison tests.

    Mirrors the pattern in test_swarm.py but with minimal attributes needed for AgentBase tests.
    """
    agent = Mock(spec=Agent)
    agent.name = name
    agent.id = f"{name}_id"
    agent.messages = []
    agent.state = AgentState()
    agent._interrupt_state = _InterruptState()
    agent._model_state = {}
    agent.tool_registry = Mock()
    agent.tool_registry.registry = {}
    agent.tool_registry.process_tools = Mock()
    agent._session_manager = None
    agent.hooks = HookRegistry()

    async def mock_stream_async(*args: Any, **kwargs: Any) -> Any:
        yield {"agent_start": True, "node": name}
        result = AgentResult(
            message={"role": "assistant", "content": [{"text": response_text}]},
            stop_reason="end_turn",
            state={},
            metrics=Mock(
                accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                accumulated_metrics={"latencyMs": 100.0},
            ),
        )
        yield {"result": result}

    agent.stream_async = Mock(side_effect=mock_stream_async)
    return agent


# --- Node creation and setup ---


def test_swarm_accepts_agentbase_implementations():
    """Test that Swarm accepts AgentBase implementations (not just Agent)."""
    agentbase1 = MockAgentBase("agentbase1")
    agentbase2 = MockAgentBase("agentbase2")

    swarm = Swarm(nodes=[agentbase1, agentbase2])

    assert len(swarm.nodes) == 2
    assert swarm.nodes["agentbase1"].executor is agentbase1
    assert swarm.nodes["agentbase2"].executor is agentbase2


def test_swarm_mixed_agent_and_agentbase():
    """Test Swarm with both Agent and AgentBase implementations."""
    agent = create_mock_agent("regular_agent")
    agentbase = MockAgentBase("agentbase_node")

    swarm = Swarm(nodes=[agent, agentbase])

    assert len(swarm.nodes) == 2
    assert "regular_agent" in swarm.nodes
    assert "agentbase_node" in swarm.nodes


def test_swarm_agentbase_without_name_attribute():
    """Test Swarm handles AgentBase without name attribute."""
    agentbase = NamelessAgentBase()

    swarm = Swarm(nodes=[agentbase])

    assert "node_0" in swarm.nodes


# --- Entry point resolution ---


def test_swarm_entry_point_named_agentbase():
    """Test that entry_point resolves correctly for a named AgentBase."""
    agentbase = MockAgentBase("my_entry")
    other = MockAgentBase("other")

    swarm = Swarm(nodes=[other, agentbase], entry_point=agentbase)

    initial = swarm._initial_node()
    assert initial.executor is agentbase
    assert initial.node_id == "my_entry"


def test_swarm_entry_point_nameless_agentbase():
    """Test that entry_point resolves correctly for a nameless AgentBase via identity lookup."""
    agentbase = NamelessAgentBase()
    named = MockAgentBase("other_agent")

    swarm = Swarm(nodes=[agentbase, named], entry_point=agentbase)

    initial = swarm._initial_node()
    assert initial.executor is agentbase


def test_swarm_entry_point_not_in_nodes_raises():
    """Test that entry_point not in nodes raises ValueError."""
    agentbase = MockAgentBase("in_swarm")
    outsider = MockAgentBase("not_in_swarm")

    with pytest.raises(ValueError, match="Entry point agent not found"):
        Swarm(nodes=[agentbase], entry_point=outsider)


# --- Tool injection ---


def test_swarm_tool_injection_skips_agentbase():
    """Test that tool injection skips non-Agent nodes."""
    agentbase = MockAgentBase("agentbase_node")
    agent = create_mock_agent("agent_with_tools")

    Swarm(nodes=[agentbase, agent])

    agent.tool_registry.process_tools.assert_called_once()
    assert not hasattr(agentbase, "tool_registry")


def test_swarm_prompt_text_conditional_on_handoff_capability():
    """Test that node input prompt reflects actual handoff capability."""
    agent = create_mock_agent("agent_node")
    agentbase = MockAgentBase("agentbase_node")

    swarm = Swarm(nodes=[agent, agentbase])

    agent_prompt = swarm._build_node_input(swarm.nodes["agent_node"])
    assert "swarm coordination tools" in agent_prompt

    agentbase_prompt = swarm._build_node_input(swarm.nodes["agentbase_node"])
    assert "swarm coordination tools" not in agentbase_prompt


def test_swarm_prompt_includes_agent_description():
    """Test that node input prompt includes description when available."""
    agent = create_mock_agent("agent_node")
    agentbase = MockAgentBase("agentbase_node")
    agentbase.description = "A helpful assistant"

    swarm = Swarm(nodes=[agent, agentbase])

    prompt = swarm._build_node_input(swarm.nodes["agent_node"])
    assert "A helpful assistant" in prompt


# --- State management ---


def test_swarm_node_state_management_with_agentbase():
    """Test SwarmNode handles missing Agent-specific attributes gracefully."""
    agentbase = MockAgentBase("agentbase_node")

    node = SwarmNode(node_id="test_node", executor=agentbase)

    assert node._initial_messages == []
    assert node._initial_state.get() == {}

    node.reset_executor_state()


# --- Interrupt handling ---


def test_swarm_interrupt_saves_context_for_agent_only():
    """Test that interrupt handling saves Agent-specific context but records all nodes."""
    agent = create_mock_agent("agent_node")
    agentbase = MockAgentBase("agentbase_node")
    swarm = Swarm(nodes=[agent, agentbase])
    interrupts = [Interrupt(id="test_id", name="test_interrupt", reason="test")]

    # Interrupt on Agent node saves full context
    swarm._activate_interrupt(swarm.nodes["agent_node"], interrupts)
    assert "agent_node" in swarm._interrupt_state.context
    assert "interrupt_state" in swarm._interrupt_state.context["agent_node"]
    assert "messages" in swarm._interrupt_state.context["agent_node"]

    # Interrupt on AgentBase node records the node but without Agent-specific fields
    swarm._activate_interrupt(swarm.nodes["agentbase_node"], interrupts)
    assert "agentbase_node" in swarm._interrupt_state.context
    assert "interrupt_state" not in swarm._interrupt_state.context["agentbase_node"]
    assert swarm._interrupt_state.activated


def test_swarm_interrupt_resume_no_keyerror_for_agentbase():
    """Test that interrupt resume handles non-Agent nodes without KeyError."""
    agent = create_mock_agent("agent_node")
    agentbase = MockAgentBase("agentbase_node")
    swarm = Swarm(nodes=[agent, agentbase])
    interrupts = [Interrupt(id="test_id", name="test_interrupt", reason="test")]

    swarm._activate_interrupt(swarm.nodes["agentbase_node"], interrupts)

    # Should not raise KeyError
    swarm.nodes["agentbase_node"].reset_executor_state()


def test_swarm_interrupt_resume_agent_without_context_returns_early():
    """Test that reset_executor_state returns early when Agent node has no saved context."""
    agent = create_mock_agent("agent_node")
    other = create_mock_agent("other_node")
    swarm = Swarm(nodes=[agent, other])

    # Manually activate interrupt state without saving context for agent_node
    swarm._interrupt_state.activate()

    # Should return early without KeyError (defensive guard)
    swarm.nodes["agent_node"].reset_executor_state()


# --- Execution ---


@pytest.mark.asyncio
async def test_swarm_execution_with_agentbase():
    """Test end-to-end Swarm execution with an AgentBase node."""
    agentbase = MockAgentBase("agentbase_node", "AgentBase completed task")

    swarm = Swarm(nodes=[agentbase])
    result = await swarm.invoke_async("Test task for AgentBase")

    assert result.status == Status.COMPLETED
    assert "agentbase_node" in result.results
    assert agentbase._call_count >= 1
