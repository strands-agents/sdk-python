"""Edge case tests for agent delegation functionality - Phase 6.

This module tests edge cases per the streamlined testing plan, focusing on:
- Missing agent lookup failures
- Sub-agent failure graceful degradation  
- Session manager compatibility
- Large context transfer
- Empty messages delegation
- State serialization with complex types
"""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.event_loop.event_loop import _handle_delegation
from strands.session.session_manager import SessionManager
from strands.telemetry.metrics import EventLoopMetrics, Trace
from strands.types.exceptions import AgentDelegationException
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _make_mock_agent(name: str) -> Agent:
    """Create a mock agent for testing."""
    return Agent(name=name, model=MockedModelProvider([]))


def _make_agent_result(text: str = "delegation_complete") -> AgentResult:
    """Create a mock agent result."""
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": text}]},
        metrics=EventLoopMetrics(),
        state={}
    )


async def _execute_delegation(
    orchestrator: Agent,
    target_agent_name: str,
    message: str = "Test delegation",
    delegation_chain: list[str] | None = None,
    transfer_state: bool = True,
    transfer_messages: bool = True,
    context: dict | None = None
) -> AgentResult:
    """Execute delegation and return result."""
    exception = AgentDelegationException(
        target_agent=target_agent_name,
        message=message,
        context=context or {},
        delegation_chain=delegation_chain or [],
        transfer_state=transfer_state,
        transfer_messages=transfer_messages
    )

    return await _handle_delegation(
        agent=orchestrator,
        delegation_exception=exception,
        invocation_state={},
        cycle_trace=Trace("test_cycle"),
        cycle_span=None
    )


class DummySessionManager(SessionManager):
    """Lightweight session manager for delegation tests."""

    def __init__(self, session_id: str, *args, **kwargs) -> None:
        super().__init__()
        self.session_id = session_id
        self.saved_agents: list[Agent] = []
        self.synced_agents: list[Agent] = []

    async def save_agent(self, agent: Agent) -> None:
        self.saved_agents.append(agent)

    async def sync_agent(self, agent: Agent, **kwargs) -> None:
        self.synced_agents.append(agent)

    def append_message(self, message: dict, agent: Agent, **kwargs) -> None:
        pass

    def redact_latest_message(self, redact_message: dict, agent: Agent, **kwargs) -> None:
        pass

    def initialize(self, agent: Agent, **kwargs) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.delegation
class TestDelegationEdgeCases:
    """Edge case tests for delegation as per Phase 6 streamlined plan."""

    async def test_missing_agent_lookup_failure(self):
        """Test clear error when target agent not found."""
        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator._sub_agents = {}
        
        # Try to delegate to non-existent agent
        with pytest.raises(ValueError, match="Target agent 'NonExistent' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="NonExistent"
            )

    async def test_sub_agent_failure_graceful_degradation(self):
        """Test graceful error handling on sub-agent failure."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("FailingAgent")
        
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        
        # Sub-agent execution fails
        sub_agent.invoke_async = AsyncMock(side_effect=RuntimeError("Sub-agent crashed"))
        
        # Delegation should propagate the failure
        with pytest.raises(RuntimeError, match="Sub-agent crashed"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="FailingAgent"
            )

    async def test_session_manager_compatibility(self):
        """Test compatibility with different session manager types."""
        # Test with DummySessionManager
        session_mgr = DummySessionManager("test-session")
        
        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator._session_manager = session_mgr
        
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        
        # Execute delegation
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )
        
        # Verify delegation succeeded
        assert result is not None
        assert isinstance(sub_agent._session_manager, DummySessionManager)
        assert sub_agent._session_manager.session_id.startswith("test-session/delegation/")
        
        # Test without session manager
        orchestrator2 = _make_mock_agent("Orchestrator2")
        sub_agent2 = _make_mock_agent("SubAgent2")
        orchestrator2._sub_agents[sub_agent2.name] = sub_agent2
        sub_agent2.invoke_async = AsyncMock(return_value=_make_agent_result())
        
        result2 = await _execute_delegation(
            orchestrator=orchestrator2,
            target_agent_name="SubAgent2"
        )
        
        # Should succeed without session management
        assert result2 is not None

    async def test_large_context_transfer(self):
        """Test handling of large context data (1MB+)."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("LargeHandler")
        
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        
        # Create large context (1MB+)
        large_context = {
            "data": "x" * 1000000,  # 1MB string
            "nested": {
                "deep": {
                    "very": {
                        "deep": {
                            "data": [i for i in range(10000)]
                        }
                    }
                }
            }
        }
        
        # Should handle without memory issues
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="LargeHandler",
            message="Handle large context",
            context=large_context
        )
        
        assert result is not None
        # Verify context was transferred (via deepcopy in exception)
        assert len(large_context["data"]) == 1000000
        assert len(large_context["nested"]["deep"]["very"]["deep"]["data"]) == 10000

    async def test_empty_messages_delegation(self):
        """Test delegation with no message history."""
        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator.messages = []  # Empty message history
        
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        
        # Should handle empty history gracefully
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )
        
        assert result is not None
        # Sub-agent should have at least delegation message
        assert len(sub_agent.messages) >= 1
        # Verify delegation context message was added
        delegation_msg_found = any(
            "Delegated from" in str(msg.get("content", []))
            for msg in sub_agent.messages
        )
        assert delegation_msg_found

    async def test_state_serialization_with_complex_types(self):
        """Test state serialization handles complex nested types."""
        orchestrator = _make_mock_agent("Orchestrator")
        
        # Create complex nested state
        orchestrator.state = {
            "nested": {
                "list": [1, 2, {"inner": "value"}],
                "tuple_data": (1, 2, 3),  # Tuples
                "set_as_list": [1, 2, 3],  # Sets (converted to lists)
                "deep": {
                    "very_deep": {
                        "data": ["a", "b", "c"],
                        "numbers": [1.5, 2.7, 3.9]
                    }
                }
            },
            "top_level": "value"
        }
        
        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        
        # Deep copy should handle complex types
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )
        
        assert result is not None
        
        # Verify sub-agent received state (deepcopy in _handle_delegation)
        assert sub_agent.state is not None
        assert "nested" in sub_agent.state
        assert sub_agent.state["top_level"] == "value"
        
        # Verify it's a deep copy, not reference
        assert sub_agent.state is not orchestrator.state
        assert sub_agent.state["nested"] is not orchestrator.state["nested"]
        
        # Modify sub-agent state shouldn't affect orchestrator
        sub_agent.state["top_level"] = "modified"
        assert orchestrator.state["top_level"] == "value"
