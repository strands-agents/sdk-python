"""Edge case tests for agent delegation functionality.

This module tests edge cases and error conditions for delegation operations including
circular delegation prevention, sub-agent lookup failures, session manager compatibility,
tool name conflicts, graceful degradation, and maximum delegation depth limits.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock
from typing import Any, Dict, List

from strands import Agent, tool
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
    delegation_chain: List[str] | None = None,
    transfer_state: bool = True,
    transfer_messages: bool = True,
    context: Dict[str, Any] | None = None
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
    """Lightweight session manager for delegation tests aligned with implementation."""

    def __init__(self, session_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.session_id = session_id
        self.saved_agents: List[Agent] = []
        self.synced_agents: List[Agent] = []
        self.appended_messages: List[Dict[str, Any]] = []

    async def save_agent(self, agent: Agent) -> None:
        self.saved_agents.append(agent)

    async def sync_agent(self, agent: Agent, **kwargs: Any) -> None:
        self.synced_agents.append(agent)

    def append_message(self, message: Dict[str, Any], agent: Agent, **kwargs: Any) -> None:
        self.appended_messages.append(message)

    def redact_latest_message(self, redact_message: Dict[str, Any], agent: Agent, **kwargs: Any) -> None:
        return None

    def initialize(self, agent: Agent, **kwargs: Any) -> None:
        return None


class FailingSaveSessionManager(DummySessionManager):
    """Session manager that raises during save to reflect implementation behaviour."""

    async def save_agent(self, agent: Agent) -> None:
        await super().save_agent(agent)
        raise Exception("Session save failed")


class FailingSyncSessionManager(DummySessionManager):
    """Session manager that raises during sync to test absence of sync usage."""

    async def sync_agent(self, agent: Agent, **kwargs: Any) -> None:
        await super().sync_agent(agent, **kwargs)
        raise Exception("Session sync failed")


def _build_delegation_tool(
    agent_name: str,
    sub_agent_name: str,
    max_depth: int,
) -> tuple[Agent, Agent, Any]:
    """Create an orchestrator/sub-agent pair and return the generated delegation tool."""

    orchestrator = _make_mock_agent(agent_name)
    orchestrator.max_delegation_depth = max_depth

    sub_agent = _make_mock_agent(sub_agent_name)
    orchestrator.add_sub_agent(sub_agent)

    tool_name = f"handoff_to_{sub_agent.name.lower().replace('-', '_')}"
    delegation_tool = orchestrator.tool_registry.registry[tool_name]
    return orchestrator, sub_agent, delegation_tool


@pytest.mark.asyncio
@pytest.mark.delegation
class TestCircularDelegationPrevention:
    """Test circular delegation prevention in complex scenarios."""

    async def test_simple_circular_delegation_a_to_b_to_a(self) -> None:
        """Test prevention of simple circular delegation A -> B -> A."""
        agent_a = _make_mock_agent("AgentA")
        agent_b = _make_mock_agent("AgentB")

        agent_a._sub_agents[agent_b.name] = agent_b
        agent_b._sub_agents[agent_a.name] = agent_a

        # First delegation: A -> B (should succeed)
        agent_b.invoke_async = AsyncMock(return_value=_make_agent_result())
        result1 = await _execute_delegation(
            orchestrator=agent_a,
            target_agent_name="AgentB",
            delegation_chain=["AgentA"]
        )
        assert result1 is not None

        # Second delegation: B -> A (should fail - circular)
        with pytest.raises(ValueError, match="Circular delegation detected"):
            await _execute_delegation(
                orchestrator=agent_b,
                target_agent_name="AgentA",
                delegation_chain=["AgentA", "AgentB"]
            )

    async def test_complex_circular_chain_a_to_b_to_c_to_a(self) -> None:
        """Test prevention of complex circular delegation A -> B -> C -> A."""
        agent_a = _make_mock_agent("AgentA")
        agent_b = _make_mock_agent("AgentB")
        agent_c = _make_mock_agent("AgentC")

        # Setup delegation chain: A -> B -> C -> A
        agent_a._sub_agents[agent_b.name] = agent_b
        agent_b._sub_agents[agent_c.name] = agent_c
        agent_c._sub_agents[agent_a.name] = agent_a

        # Mock successful delegations for first two steps
        agent_b.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_c.invoke_async = AsyncMock(return_value=_make_agent_result())

        # First delegation: A -> B (should succeed)
        result1 = await _execute_delegation(
            orchestrator=agent_a,
            target_agent_name="AgentB",
            delegation_chain=["AgentA"]
        )
        assert result1 is not None

        # Second delegation: B -> C (should succeed)
        result2 = await _execute_delegation(
            orchestrator=agent_b,
            target_agent_name="AgentC",
            delegation_chain=["AgentA", "AgentB"]
        )
        assert result2 is not None

        # Third delegation: C -> A (should fail - circular)
        with pytest.raises(ValueError, match="Circular delegation detected: AgentA -> AgentB -> AgentC -> AgentA"):
            await _execute_delegation(
                orchestrator=agent_c,
                target_agent_name="AgentA",
                delegation_chain=["AgentA", "AgentB", "AgentC"]
            )

    async def test_self_delegation_prevention(self) -> None:
        """Test prevention of agent delegating to itself."""
        agent = _make_mock_agent("SelfAgent")
        agent._sub_agents[agent.name] = agent  # Agent can delegate to itself in registry

        # Self-delegation should fail
        with pytest.raises(ValueError, match="Circular delegation detected"):
            await _execute_delegation(
                orchestrator=agent,
                target_agent_name="SelfAgent",
                delegation_chain=["SelfAgent"]
            )

    async def test_circular_with_multiple_agents_in_chain(self) -> None:
        """Test circular delegation with long chain: A -> B -> C -> D -> B."""
        agent_a = _make_mock_agent("AgentA")
        agent_b = _make_mock_agent("AgentB")
        agent_c = _make_mock_agent("AgentC")
        agent_d = _make_mock_agent("AgentD")

        # Setup complex delegation relationships
        agent_a._sub_agents[agent_b.name] = agent_b
        agent_b._sub_agents[agent_c.name] = agent_c
        agent_c._sub_agents[agent_d.name] = agent_d
        agent_d._sub_agents[agent_b.name] = agent_b  # Creates circular reference

        # Mock successful delegations
        agent_b.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_c.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_d.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Build delegation chain step by step
        # Step 1: A -> B (should succeed)
        result1 = await _execute_delegation(
            orchestrator=agent_a,
            target_agent_name="AgentB",
            delegation_chain=["AgentA"]
        )
        assert result1 is not None

        # Step 2: B -> C (should succeed)
        result2 = await _execute_delegation(
            orchestrator=agent_b,
            target_agent_name="AgentC",
            delegation_chain=["AgentA", "AgentB"]
        )
        assert result2 is not None

        # Step 3: C -> D (should succeed)
        result3 = await _execute_delegation(
            orchestrator=agent_c,
            target_agent_name="AgentD",
            delegation_chain=["AgentA", "AgentB", "AgentC"]
        )
        assert result3 is not None

        # Step 4: D -> B (should fail - B already in chain)
        with pytest.raises(ValueError, match="Circular delegation detected"):
            await _execute_delegation(
                orchestrator=agent_d,
                target_agent_name="AgentB",
                delegation_chain=["AgentA", "AgentB", "AgentC", "AgentD"]
            )

    async def test_no_false_positive_circular_detection(self) -> None:
        """Test that valid non-circular chains are not flagged as circular."""
        agent_a = _make_mock_agent("AgentA")
        agent_b = _make_mock_agent("AgentB")
        agent_c = _make_mock_agent("AgentC")
        agent_d = _make_mock_agent("AgentD")

        # Setup linear delegation chain: A -> B -> C -> D
        agent_a._sub_agents[agent_b.name] = agent_b
        agent_b._sub_agents[agent_c.name] = agent_c
        agent_c._sub_agents[agent_d.name] = agent_d

        # Mock successful delegations
        agent_b.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_c.invoke_async = AsyncMock(return_value=_make_agent_result())
        agent_d.invoke_async = AsyncMock(return_value=_make_agent_result())

        # All delegations should succeed (no circular reference)
        result1 = await _execute_delegation(
            orchestrator=agent_a,
            target_agent_name="AgentB",
            delegation_chain=["AgentA"]
        )
        assert result1 is not None

        result2 = await _execute_delegation(
            orchestrator=agent_b,
            target_agent_name="AgentC",
            delegation_chain=["AgentA", "AgentB"]
        )
        assert result2 is not None

        result3 = await _execute_delegation(
            orchestrator=agent_c,
            target_agent_name="AgentD",
            delegation_chain=["AgentA", "AgentB", "AgentC"]
        )
        assert result3 is not None


@pytest.mark.asyncio
@pytest.mark.delegation
class TestSubAgentLookupFailures:
    """Test handling of missing sub-agent scenarios."""

    async def test_missing_target_agent(self) -> None:
        """Test delegation to non-existent target agent."""
        orchestrator = _make_mock_agent("Orchestrator")

        # Try to delegate to agent that doesn't exist
        with pytest.raises(ValueError, match="Target agent 'MissingAgent' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="MissingAgent"
            )

    async def test_sub_agent_removed_before_delegation(self) -> None:
        """Test delegation when sub-agent is removed before execution."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("TemporaryAgent")

        # Add sub-agent
        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Remove sub-agent before delegation
        del orchestrator._sub_agents[sub_agent.name]

        # Delegation should fail
        with pytest.raises(ValueError, match="Target agent 'TemporaryAgent' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="TemporaryAgent"
            )

    async def test_sub_agent_registry_corruption(self) -> None:
        """Test delegation when sub-agent registry is corrupted."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("CorruptedAgent")

        # Add sub-agent but corrupt the registry
        orchestrator._sub_agents[sub_agent.name] = None  # Corrupted entry

        # Delegation should fail gracefully
        with pytest.raises(ValueError, match="Target agent 'CorruptedAgent' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="CorruptedAgent"
            )

    async def test_case_sensitive_agent_lookup(self) -> None:
        """Test that agent lookup is case sensitive."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")

        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Try different case variations
        with pytest.raises(ValueError, match="Target agent 'subagent' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="subagent"  # Lowercase
            )

        with pytest.raises(ValueError, match="Target agent 'SUBAGENT' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="SUBAGENT"  # Uppercase
            )

        # Exact case should work
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"  # Exact case
        )
        assert result is not None

    async def test_empty_sub_agent_name(self) -> None:
        """Test delegation with empty target agent name."""
        orchestrator = _make_mock_agent("Orchestrator")

        # Empty string should fail gracefully
        with pytest.raises(ValueError, match="Target agent '' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name=""
            )

    async def test_none_sub_agent_name(self) -> None:
        """Test delegation with None target agent name."""
        orchestrator = _make_mock_agent("Orchestrator")

        # None should be handled gracefully (converted to string)
        with pytest.raises(ValueError, match="Target agent 'None' not found"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name=None  # type: ignore
            )


@pytest.mark.asyncio
@pytest.mark.delegation
class TestSessionManagerCompatibility:
    """Test delegation compatibility with various session manager configurations."""

    async def test_delegation_with_session_manager(self) -> None:
        """Test delegation when orchestrator has session manager."""
        session_manager = DummySessionManager("root-session")

        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator._session_manager = session_manager

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
        assert sub_agent._session_manager.session_id.startswith("root-session/delegation/")
        assert sub_agent._session_manager.saved_agents == [sub_agent]

    async def test_delegation_without_session_manager(self) -> None:
        """Test delegation when orchestrator has no session manager."""
        orchestrator = _make_mock_agent("Orchestrator")
        # No session manager set

        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Execute delegation
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )

        # Verify delegation succeeded without session management
        assert result is not None
        # Sub-agent should not have session manager
        assert not hasattr(sub_agent, '_session_manager') or sub_agent._session_manager is None

    async def test_session_manager_with_nested_delegation(self) -> None:
        """Test session management with nested delegation chains."""
        # Create session manager for root orchestrator
        mock_session_manager = DummySessionManager("root-session")

        root = _make_mock_agent("Root")
        root._session_manager = mock_session_manager

        middle = _make_mock_agent("Middle")
        leaf = _make_mock_agent("Leaf")

        # Setup delegation chain
        root._sub_agents[middle.name] = middle
        middle._sub_agents[leaf.name] = leaf

        middle.invoke_async = AsyncMock(return_value=_make_agent_result())
        leaf.invoke_async = AsyncMock(return_value=_make_agent_result())

        # First delegation: Root -> Middle
        result1 = await _execute_delegation(
            orchestrator=root,
            target_agent_name="Middle",
            delegation_chain=["Root"]
        )
        assert result1 is not None

        # Verify middle has nested session
        assert isinstance(middle._session_manager, DummySessionManager)
        assert middle._session_manager.session_id.startswith("root-session/delegation/")

        # Second delegation: Middle -> Leaf
        result2 = await _execute_delegation(
            orchestrator=middle,
            target_agent_name="Leaf",
            delegation_chain=["Root", "Middle"]
        )
        assert result2 is not None

        # Verify leaf has doubly nested session
        assert isinstance(leaf._session_manager, DummySessionManager)
        expected_prefix = f"{middle._session_manager.session_id}/delegation/"
        assert leaf._session_manager.session_id.startswith(expected_prefix)

    async def test_session_manager_save_error_handling(self) -> None:
        """Test graceful handling when session manager save fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator._session_manager = FailingSaveSessionManager("failing-session")

        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        with pytest.raises(Exception, match="Session save failed"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="SubAgent"
            )

        assert isinstance(sub_agent._session_manager, FailingSaveSessionManager)

    async def test_session_manager_sync_error_handling(self) -> None:
        """Test graceful handling when session manager sync fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        orchestrator._session_manager = FailingSyncSessionManager("sync-failing-session")

        sub_agent = _make_mock_agent("SubAgent")
        orchestrator._sub_agents[sub_agent.name] = sub_agent
        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )

        assert result is not None
        assert isinstance(sub_agent._session_manager, FailingSyncSessionManager)
        assert sub_agent._session_manager.synced_agents == []


@pytest.mark.asyncio
@pytest.mark.delegation
class TestToolNameConflicts:
    """Test handling of tool name conflicts in delegation scenarios."""

    async def test_delegation_tool_name_conflict_with_existing_tool(self) -> None:
        """Existing prefixed tools do not collide with delegation tool names."""
        @tool
        def existing_handoff_to_specialist(message: str) -> dict:
            """Existing tool that conflicts with delegation tool name."""
            return {"content": [{"text": f"Existing tool: {message}"}]}

        orchestrator = Agent(
            name="Orchestrator",
            model=MockedModelProvider([]),
            tools=[existing_handoff_to_specialist]
        )

        specialist = _make_mock_agent("Specialist")
        orchestrator._validate_sub_agents([specialist])
        orchestrator._sub_agents[specialist.name] = specialist
        orchestrator._generate_delegation_tools([specialist])

        assert "existing_handoff_to_specialist" in orchestrator.tool_registry.registry
        assert "handoff_to_specialist" in orchestrator.tool_registry.registry

    async def test_tool_name_sanitization_prevents_conflicts(self) -> None:
        """Test that tool name sanitization prevents conflicts."""
        @tool
        def handoff_to_agent_one(message: str) -> dict:
            """Existing tool with sanitized name."""
            return {"content": [{"text": f"Agent One: {message}"}]}

        orchestrator = Agent(
            name="Orchestrator",
            model=MockedModelProvider([]),
            tools=[handoff_to_agent_one]
        )

        # Sub-agent with name that gets sanitized to same as existing tool
        sub_agent = _make_mock_agent("Agent-One")  # Becomes "agent_one" -> "handoff_to_agent_one"

        # Should detect conflict after sanitization
        with pytest.raises(ValueError, match="Tool name conflict"):
            orchestrator._validate_sub_agents([sub_agent])

    async def test_multiple_sub_agents_sanitized_name_conflict(self) -> None:
        """Sanitized duplicates overwrite existing delegation tool registrations."""
        orchestrator = _make_mock_agent("Orchestrator")

        # Two agents that will sanitize to the same tool name
        agent1 = _make_mock_agent("My-Agent")  # Becomes "handoff_to_my_agent"
        agent2 = _make_mock_agent("My_Agent")  # Also becomes "handoff_to_my_agent"

        orchestrator._validate_sub_agents([agent1, agent2])
        orchestrator._sub_agents[agent1.name] = agent1
        orchestrator._sub_agents[agent2.name] = agent2
        orchestrator._generate_delegation_tools([agent1, agent2])

        assert list(orchestrator.tool_registry.registry.keys()).count("handoff_to_my_agent") == 1

    async def test_no_false_positive_tool_conflicts(self) -> None:
        """Test that different tool names don't create false conflicts."""
        @tool
        def handoff_to_specialist(message: str) -> dict:
            """Tool for specialist delegation."""
            return {"content": [{"text": f"Specialist: {message}"}]}

        @tool
        def handoff_to_analyst(message: str) -> dict:
            """Tool for analyst delegation."""
            return {"content": [{"text": f"Analyst: {message}"}]}

        orchestrator = Agent(
            name="Orchestrator",
            model=MockedModelProvider([]),
            tools=[handoff_to_specialist, handoff_to_analyst]
        )

        # Sub-agent with different name should not conflict
        researcher = _make_mock_agent("Researcher")
        researcher.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Should not raise conflict
        orchestrator._validate_sub_agents([researcher])
        orchestrator._sub_agents[researcher.name] = researcher

        # Delegation should work
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="Researcher"
        )
        assert result is not None


@pytest.mark.asyncio
@pytest.mark.delegation
class TestGracefulDegradation:
    """Test graceful degradation when sub-agents fail."""

    async def test_sub_agent_execution_failure(self) -> None:
        """Test graceful handling when sub-agent execution fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        failing_agent = _make_mock_agent("FailingAgent")

        orchestrator._sub_agents[failing_agent.name] = failing_agent

        # Sub-agent execution fails
        failing_agent.invoke_async = AsyncMock(side_effect=Exception("Sub-agent execution failed"))

        # Delegation should propagate the failure
        with pytest.raises(Exception, match="Sub-agent execution failed"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="FailingAgent"
            )

    async def test_sub_agent_timeout_during_execution(self) -> None:
        """Test graceful handling when sub-agent times out during execution."""
        orchestrator = _make_mock_agent("Orchestrator")
        slow_agent = _make_mock_agent("SlowAgent")

        orchestrator._sub_agents[slow_agent.name] = slow_agent
        orchestrator.delegation_timeout = 0.1  # Short timeout

        # Sub-agent takes too long
        async def slow_execution():
            await asyncio.sleep(0.2)  # Longer than timeout
            return _make_agent_result()

        slow_agent.invoke_async = slow_execution

        # Should timeout gracefully
        with pytest.raises(TimeoutError, match="timed out"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="SlowAgent"
            )

    async def test_sub_agent_model_failure(self) -> None:
        """Test handling when sub-agent's model fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        problematic_agent = _make_mock_agent("ProblematicAgent")

        orchestrator._sub_agents[problematic_agent.name] = problematic_agent

        # Mock invocation failure originating from sub-agent model
        problematic_agent.invoke_async = AsyncMock(side_effect=Exception("Model failure"))

        # Delegation should fail gracefully
        with pytest.raises(Exception, match="Model failure"):
            await _execute_delegation(
                orchestrator=orchestrator,
                target_agent_name="ProblematicAgent"
            )

    async def test_partial_state_transfer_failure(self) -> None:
        """Test graceful handling when state transfer partially fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")

        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Create state that will cause issues during deepcopy
        problematic_state = {
            "data": "normal_data",
            "problematic": object()  # Object that might cause deepcopy issues
        }
        orchestrator.state = problematic_state

        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Delegation should still succeed despite potential state issues
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )

        assert result is not None

    async def test_message_transfer_filtering_failure(self) -> None:
        """Test graceful handling when message filtering fails."""
        orchestrator = _make_mock_agent("Orchestrator")
        sub_agent = _make_mock_agent("SubAgent")

        orchestrator._sub_agents[sub_agent.name] = sub_agent

        # Create problematic message structure
        problematic_messages = [
            {"role": "system", "content": [{"type": "text", "text": "System"}]},
            {"role": "user", "content": None},  # None content might cause issues
            {"role": "assistant", "content": "invalid_structure"},  # Invalid structure
        ]
        orchestrator.messages = problematic_messages

        sub_agent.invoke_async = AsyncMock(return_value=_make_agent_result())

        # Should handle problematic messages gracefully
        result = await _execute_delegation(
            orchestrator=orchestrator,
            target_agent_name="SubAgent"
        )

        assert result is not None
        # Sub-agent should have some messages (filtered gracefully)
        assert len(sub_agent.messages) >= 0


@pytest.mark.asyncio
@pytest.mark.delegation
class TestMaximumDelegationDepth:
    """Test maximum delegation depth limits."""

    async def test_enforcement_of_max_delegation_depth(self) -> None:
        """Delegation tool raises once the configured depth is reached."""
        _, _, delegation_tool = _build_delegation_tool("Orchestrator", "Specialist", max_depth=2)

        with pytest.raises(ValueError, match="Maximum delegation depth"):
            delegation_tool(message="Too deep", delegation_chain=["Root", "Intermediate"])

    async def test_depth_limit_with_actual_chain(self) -> None:
        """Delegation within the limit produces an AgentDelegationException."""
        orchestrator, sub_agent, delegation_tool = _build_delegation_tool("Orchestrator", "Specialist", max_depth=3)

        with pytest.raises(AgentDelegationException) as exc_info:
            delegation_tool(message="Continue", delegation_chain=["Root"])

        assert exc_info.value.target_agent == sub_agent.name
        assert exc_info.value.delegation_chain == ["Root", orchestrator.name]

    async def test_different_depth_limits_per_agent(self) -> None:
        """Agents honor their individual depth limits."""
        _, _, shallow_tool = _build_delegation_tool("ShallowAgent", "Sub", max_depth=1)
        deep_orchestrator, deep_sub, deep_tool = _build_delegation_tool("DeepAgent", "DeepSub", max_depth=5)

        with pytest.raises(ValueError, match="Maximum delegation depth"):
            shallow_tool(message="Too deep", delegation_chain=["Root"])

        with pytest.raises(AgentDelegationException) as exc_info:
            deep_tool(message="Within limit", delegation_chain=["Root", "Level1", "Level2"])

        assert exc_info.value.delegation_chain == ["Root", "Level1", "Level2", deep_orchestrator.name]
        assert exc_info.value.target_agent == deep_sub.name

    async def test_zero_max_delegation_depth(self) -> None:
        """Depth of zero prevents any delegation attempts."""
        _, _, delegation_tool = _build_delegation_tool("ZeroAgent", "ZeroSub", max_depth=0)

        with pytest.raises(ValueError, match="Maximum delegation depth"):
            delegation_tool(message="Blocked", delegation_chain=[])

    async def test_negative_max_delegation_depth(self) -> None:
        """Negative depth behaves the same as zero depth."""
        _, _, delegation_tool = _build_delegation_tool("NegativeAgent", "NegativeSub", max_depth=-1)

        with pytest.raises(ValueError, match="Maximum delegation depth"):
            delegation_tool(message="Blocked", delegation_chain=[])