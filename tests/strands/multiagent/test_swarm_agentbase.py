"""Unit tests for Swarm AgentBase protocol support.

Tests that Swarm correctly handles AgentBase implementations that are not
concrete Agent instances, following the same patterns as Graph.
"""

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from strands.agent import Agent, AgentResult
from strands.multiagent.base import Status
from strands.multiagent.swarm import Swarm, SwarmNode


class MockAgentBase:
    """Mock implementation of AgentBase protocol for testing.

    This implementation only provides the minimal AgentBase interface
    without Agent-specific attributes like messages, state, tool_registry, etc.
    """

    def __init__(self, name: str, response_text: str = "AgentBase response"):
        """Initialize mock AgentBase.

        Args:
            name: Name of the agent
            response_text: Text to return in responses
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
        yield {"agent_thinking": True, "thought": f"Processing with {self.name}"}
        result = await self.invoke_async(prompt, **kwargs)
        yield {"result": result}


def create_mock_agent(name: str, response_text: str = "Agent response") -> Agent:
    """Create a mock Agent for comparison tests."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.id = f"{name}_id"
    agent.messages = []
    agent.state = Mock()
    agent.state.get = Mock(return_value={})
    agent._interrupt_state = Mock()
    agent._interrupt_state.to_dict = Mock(return_value={})
    agent.tool_registry = Mock()
    agent.tool_registry.registry = {}
    agent.tool_registry.process_tools = Mock()
    agent._session_manager = None

    async def mock_stream_async(*args, **kwargs):
        yield {"agent_start": True, "node": name}
        yield {"agent_thinking": True, "thought": f"Processing with {name}"}
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


def test_swarm_accepts_agentbase_implementations():
    """Test that Swarm accepts AgentBase implementations (not just Agent)."""
    # Create AgentBase implementations
    agentbase1 = MockAgentBase("agentbase1", "AgentBase 1 response")
    agentbase2 = MockAgentBase("agentbase2", "AgentBase 2 response")

    # Should not raise any errors
    swarm = Swarm(nodes=[agentbase1, agentbase2])

    # Verify nodes were created
    assert len(swarm.nodes) == 2
    assert "agentbase1" in swarm.nodes
    assert "agentbase2" in swarm.nodes

    # Verify node executors are the AgentBase implementations
    assert swarm.nodes["agentbase1"].executor is agentbase1
    assert swarm.nodes["agentbase2"].executor is agentbase2


def test_swarm_mixed_agent_and_agentbase():
    """Test Swarm with both Agent and AgentBase implementations."""
    # Create mix of Agent and AgentBase
    agent = create_mock_agent("regular_agent")
    agentbase = MockAgentBase("agentbase_node")

    # Should work with mixed types
    swarm = Swarm(nodes=[agent, agentbase])

    assert len(swarm.nodes) == 2
    assert "regular_agent" in swarm.nodes
    assert "agentbase_node" in swarm.nodes


def test_swarm_tool_injection_skips_agentbase():
    """Test that tool injection gracefully skips non-Agent nodes."""
    # Create AgentBase without tool_registry
    agentbase = MockAgentBase("agentbase_node")
    agent = create_mock_agent("agent_with_tools")

    # Create swarm - should not fail
    Swarm(nodes=[agentbase, agent])

    # Verify tool injection was attempted only on Agent
    agent.tool_registry.process_tools.assert_called_once()

    # AgentBase doesn't have tool_registry, so it should be skipped
    assert not hasattr(agentbase, "tool_registry")


def test_swarm_node_state_management_with_agentbase():
    """Test SwarmNode state management with AgentBase (hasattr checks)."""
    # Create AgentBase without messages/state attributes
    agentbase = MockAgentBase("agentbase_node")

    # Create SwarmNode - should not fail
    node = SwarmNode(node_id="test_node", executor=agentbase)

    # __post_init__ should handle missing attributes gracefully
    assert node._initial_messages == []  # Default empty list
    assert node._initial_state.get() == {}  # Default empty state

    # reset_executor_state should not fail on AgentBase
    node.reset_executor_state()  # Should complete without error


@pytest.mark.asyncio
async def test_swarm_execution_with_agentbase():
    """Test Swarm execution with AgentBase implementations."""
    # Create AgentBase implementation
    agentbase = MockAgentBase("agentbase_node", "AgentBase completed task")

    # Create swarm
    swarm = Swarm(nodes=[agentbase])

    # Execute swarm
    result = await swarm.invoke_async("Test task for AgentBase")

    # Verify execution completed
    assert result.status == Status.COMPLETED
    assert len(result.results) == 1
    assert "agentbase_node" in result.results

    # Verify AgentBase was called
    assert agentbase._call_count >= 1


def test_swarm_agentbase_without_name_attribute():
    """Test Swarm handles AgentBase without name attribute."""

    class AgentBaseNoName:
        """AgentBase without name attribute."""

        async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
            return AgentResult(
                message={"role": "assistant", "content": [{"text": "response"}]},
                stop_reason="end_turn",
                state={},
                metrics=None,
            )

        def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
            return asyncio.run(self.invoke_async(prompt, **kwargs))

        async def stream_async(self, prompt: Any = None, **kwargs: Any):
            result = await self.invoke_async(prompt, **kwargs)
            yield {"result": result}

    # Create instance without name
    agentbase = AgentBaseNoName()

    # Should auto-generate node_id
    swarm = Swarm(nodes=[agentbase])

    # Should have generated node_0 since no name attribute
    assert "node_0" in swarm.nodes


def test_swarm_interrupt_handling_with_agentbase():
    """Test that interrupt handling only saves Agent-specific context."""
    from strands.interrupt import Interrupt

    # Create mixed nodes
    agent = create_mock_agent("agent_node")
    agentbase = MockAgentBase("agentbase_node")

    # Create swarm
    swarm = Swarm(nodes=[agent, agentbase])

    # Create interrupts
    interrupts = [Interrupt(id="test_id", name="test_interrupt", reason="test")]

    # Activate interrupt on Agent node
    agent_node = swarm.nodes["agent_node"]
    swarm._activate_interrupt(agent_node, interrupts)

    # Should have saved Agent-specific context
    assert "agent_node" in swarm._interrupt_state.context
    assert "interrupt_state" in swarm._interrupt_state.context["agent_node"]
    assert "state" in swarm._interrupt_state.context["agent_node"]
    assert "messages" in swarm._interrupt_state.context["agent_node"]

    # Activate interrupt on AgentBase node
    agentbase_node = swarm.nodes["agentbase_node"]
    swarm._activate_interrupt(agentbase_node, interrupts)

    # Should NOT have saved AgentBase context (isinstance check should prevent it)
    # The interrupt is registered but no Agent-specific context is saved
    assert swarm._interrupt_state.activated
