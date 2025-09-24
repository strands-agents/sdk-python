import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from strands.agent import Agent, AgentResult
from strands.agent.state import AgentState
from strands.hooks.registry import HookRegistry
from strands.multiagent.base import Status
from strands.multiagent.swarm import SharedContext, Swarm, SwarmNode, SwarmResult, SwarmState
from strands.session.session_manager import SessionManager
from strands.types.content import ContentBlock


def create_mock_agent(name, response_text="Default response", metrics=None, agent_id=None, should_fail=False):
    """Create a mock Agent with specified properties."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.id = agent_id or f"{name}_id"
    agent.messages = []
    agent.state = AgentState()  # Add state attribute
    agent.tool_registry = Mock()
    agent.tool_registry.registry = {}
    agent.tool_registry.process_tools = Mock()
    agent._call_count = 0
    agent._should_fail = should_fail
    agent._session_manager = None
    agent.hooks = HookRegistry()

    if metrics is None:
        metrics = Mock(
            accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            accumulated_metrics={"latencyMs": 100.0},
        )

    def create_mock_result():
        agent._call_count += 1

        # Simulate failure if requested
        if agent._should_fail:
            raise Exception("Simulated agent failure")

        return AgentResult(
            message={"role": "assistant", "content": [{"text": response_text}]},
            stop_reason="end_turn",
            state={},
            metrics=metrics,
        )

    agent.return_value = create_mock_result()
    agent.__call__ = Mock(side_effect=create_mock_result)

    async def mock_invoke_async(*args, **kwargs):
        return create_mock_result()

    agent.invoke_async = MagicMock(side_effect=mock_invoke_async)

    return agent


@pytest.fixture
def mock_agents():
    """Create a set of mock agents for testing."""
    return {
        "coordinator": create_mock_agent("coordinator", "Coordinating task"),
        "specialist": create_mock_agent("specialist", "Specialized response"),
        "reviewer": create_mock_agent("reviewer", "Review complete"),
    }


@pytest.fixture
def mock_swarm(mock_agents):
    """Create a swarm for testing."""
    agents = list(mock_agents.values())
    swarm = Swarm(
        agents,
        max_handoffs=5,
        max_iterations=5,
        execution_timeout=30.0,
        node_timeout=10.0,
    )

    return swarm


@pytest.fixture
def mock_strands_tracer():
    with patch("strands.multiagent.swarm.get_tracer") as mock_get_tracer:
        mock_tracer_instance = MagicMock()
        mock_span = MagicMock()
        mock_tracer_instance.start_multiagent_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer_instance
        yield mock_tracer_instance


@pytest.fixture
def mock_use_span():
    with patch("strands.multiagent.swarm.trace_api.use_span") as mock_use_span:
        yield mock_use_span


def test_swarm_structure_and_nodes(mock_swarm, mock_agents):
    """Test swarm structure and SwarmNode properties."""
    # Test swarm structure
    assert len(mock_swarm.nodes) == 3
    assert "coordinator" in mock_swarm.nodes
    assert "specialist" in mock_swarm.nodes
    assert "reviewer" in mock_swarm.nodes

    # Test SwarmNode properties
    coordinator_node = mock_swarm.nodes["coordinator"]
    assert coordinator_node.node_id == "coordinator"
    assert coordinator_node.executor == mock_agents["coordinator"]
    assert str(coordinator_node) == "coordinator"
    assert repr(coordinator_node) == "SwarmNode(node_id='coordinator')"

    # Test SwarmNode equality and hashing
    other_coordinator = SwarmNode("coordinator", mock_agents["coordinator"])
    assert coordinator_node == other_coordinator
    assert hash(coordinator_node) == hash(other_coordinator)
    assert coordinator_node != mock_swarm.nodes["specialist"]
    # Test SwarmNode inequality with different types
    assert coordinator_node != "not_a_swarm_node"
    assert coordinator_node != 42


def test_shared_context(mock_swarm):
    """Test SharedContext functionality and validation."""
    coordinator_node = mock_swarm.nodes["coordinator"]
    specialist_node = mock_swarm.nodes["specialist"]

    # Test SharedContext with multiple nodes (covers new node path)
    shared_context = SharedContext()
    shared_context.add_context(coordinator_node, "task_status", "in_progress")
    assert shared_context.context["coordinator"]["task_status"] == "in_progress"

    # Add context for a different node (this will create new node entry)
    shared_context.add_context(specialist_node, "analysis", "complete")
    assert shared_context.context["specialist"]["analysis"] == "complete"
    assert len(shared_context.context) == 2  # Two nodes now have context

    # Test SharedContext validation
    with pytest.raises(ValueError, match="Key cannot be None"):
        shared_context.add_context(coordinator_node, None, "value")

    with pytest.raises(ValueError, match="Key must be a string"):
        shared_context.add_context(coordinator_node, 123, "value")

    with pytest.raises(ValueError, match="Key cannot be empty"):
        shared_context.add_context(coordinator_node, "", "value")

    with pytest.raises(ValueError, match="Value is not JSON serializable"):
        shared_context.add_context(coordinator_node, "key", lambda x: x)


def test_swarm_state_should_continue(mock_swarm):
    """Test SwarmState should_continue method with various scenarios."""
    coordinator_node = mock_swarm.nodes["coordinator"]
    specialist_node = mock_swarm.nodes["specialist"]
    state = SwarmState(current_node=coordinator_node, task="test task")

    # Test normal continuation
    should_continue, reason = state.should_continue(
        max_handoffs=10,
        max_iterations=10,
        execution_timeout=60.0,
        repetitive_handoff_detection_window=0,
        repetitive_handoff_min_unique_agents=0,
    )
    assert should_continue is True
    assert reason == "Continuing"

    # Test max handoffs limit
    state.node_history = [coordinator_node] * 5
    should_continue, reason = state.should_continue(
        max_handoffs=3,
        max_iterations=10,
        execution_timeout=60.0,
        repetitive_handoff_detection_window=0,
        repetitive_handoff_min_unique_agents=0,
    )
    assert should_continue is False
    assert "Max handoffs reached" in reason

    # Test max iterations limit
    should_continue, reason = state.should_continue(
        max_handoffs=10,
        max_iterations=3,
        execution_timeout=60.0,
        repetitive_handoff_detection_window=0,
        repetitive_handoff_min_unique_agents=0,
    )
    assert should_continue is False
    assert "Max iterations reached" in reason

    # Test timeout
    state.start_time = time.time() - 100  # Set start time to 100 seconds ago
    should_continue, reason = state.should_continue(
        max_handoffs=10,
        max_iterations=10,
        execution_timeout=50.0,  # 50 second timeout
        repetitive_handoff_detection_window=0,
        repetitive_handoff_min_unique_agents=0,
    )
    assert should_continue is False
    assert "Execution timed out" in reason

    # Test repetitive handoff detection
    state.node_history = [coordinator_node, specialist_node, coordinator_node, specialist_node]
    state.start_time = time.time()  # Reset start time
    should_continue, reason = state.should_continue(
        max_handoffs=10,
        max_iterations=10,
        execution_timeout=60.0,
        repetitive_handoff_detection_window=4,
        repetitive_handoff_min_unique_agents=3,
    )
    assert should_continue is False
    assert "Repetitive handoff" in reason


@pytest.mark.asyncio
async def test_swarm_execution_async(mock_strands_tracer, mock_use_span, mock_swarm, mock_agents):
    """Test asynchronous swarm execution."""
    # Execute swarm
    task = [ContentBlock(text="Analyze this task"), ContentBlock(text="Additional context")]
    result = await mock_swarm.invoke_async(task)

    # Verify execution results
    assert result.status == Status.COMPLETED
    assert result.execution_count == 1
    assert len(result.results) == 1

    # Verify agent was called
    mock_agents["coordinator"].invoke_async.assert_called()

    # Verify metrics aggregation
    assert result.accumulated_usage["totalTokens"] >= 0
    assert result.accumulated_metrics["latencyMs"] >= 0

    # Verify result type
    assert isinstance(result, SwarmResult)
    assert hasattr(result, "node_history")
    assert len(result.node_history) == 1

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


def test_swarm_synchronous_execution(mock_strands_tracer, mock_use_span, mock_agents):
    """Test synchronous swarm execution using __call__ method."""
    agents = list(mock_agents.values())
    swarm = Swarm(
        nodes=agents,
        max_handoffs=3,
        max_iterations=3,
        execution_timeout=15.0,
        node_timeout=5.0,
    )

    # Test synchronous execution
    result = swarm("Test synchronous swarm execution")

    # Verify execution results
    assert result.status == Status.COMPLETED
    assert result.execution_count == 1
    assert len(result.results) == 1
    assert result.execution_time >= 0

    # Verify agent was called
    mock_agents["coordinator"].invoke_async.assert_called()

    # Verify return type is SwarmResult
    assert isinstance(result, SwarmResult)
    assert hasattr(result, "node_history")

    # Test swarm configuration
    assert swarm.max_handoffs == 3
    assert swarm.max_iterations == 3
    assert swarm.execution_timeout == 15.0
    assert swarm.node_timeout == 5.0

    # Test tool injection
    for node in swarm.nodes.values():
        node.executor.tool_registry.process_tools.assert_called()

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


def test_swarm_builder_validation(mock_agents):
    """Test swarm builder validation and error handling."""
    # Test agent name assignment
    unnamed_agent = create_mock_agent(None)
    unnamed_agent.name = None
    agents_with_unnamed = [unnamed_agent, mock_agents["coordinator"]]

    swarm_with_unnamed = Swarm(nodes=agents_with_unnamed)
    assert "node_0" in swarm_with_unnamed.nodes
    assert "coordinator" in swarm_with_unnamed.nodes

    # Test duplicate node names
    duplicate_agent = create_mock_agent("coordinator")
    with pytest.raises(ValueError, match="Node ID 'coordinator' is not unique"):
        Swarm(nodes=[mock_agents["coordinator"], duplicate_agent])

    # Test duplicate agent instances
    same_agent = mock_agents["coordinator"]
    with pytest.raises(ValueError, match="Duplicate node instance detected"):
        Swarm(nodes=[same_agent, same_agent])

    # Test tool name conflicts - handoff tool
    conflicting_agent = create_mock_agent("conflicting")
    conflicting_agent.tool_registry.registry = {"handoff_to_agent": Mock()}

    with pytest.raises(ValueError, match="already has tools with names that conflict"):
        Swarm(nodes=[conflicting_agent])


def test_swarm_handoff_functionality():
    """Test swarm handoff functionality."""

    # Create an agent that will hand off to another agent
    def create_handoff_agent(name, target_agent_name, response_text="Handing off"):
        """Create a mock agent that performs handoffs."""
        agent = create_mock_agent(name, response_text)
        agent._handoff_done = False  # Track if handoff has been performed

        def create_handoff_result():
            agent._call_count += 1
            # Perform handoff on first execution call (not setup calls)
            if (
                not agent._handoff_done
                and hasattr(agent, "_swarm_ref")
                and agent._swarm_ref
                and hasattr(agent._swarm_ref.state, "completion_status")
            ):
                target_node = agent._swarm_ref.nodes.get(target_agent_name)
                if target_node:
                    agent._swarm_ref._handle_handoff(
                        target_node, f"Handing off to {target_agent_name}", {"handoff_context": "test_data"}
                    )
                    agent._handoff_done = True

            return AgentResult(
                message={"role": "assistant", "content": [{"text": response_text}]},
                stop_reason="end_turn",
                state={},
                metrics=Mock(
                    accumulated_usage={"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
                    accumulated_metrics={"latencyMs": 50.0},
                ),
            )

        agent.return_value = create_handoff_result()
        agent.__call__ = Mock(side_effect=create_handoff_result)

        async def mock_invoke_async(*args, **kwargs):
            return create_handoff_result()

        agent.invoke_async = MagicMock(side_effect=mock_invoke_async)
        return agent

    # Create agents - first one hands off, second one completes by not handing off
    handoff_agent = create_handoff_agent("handoff_agent", "completion_agent")
    completion_agent = create_mock_agent("completion_agent", "Task completed")

    # Create a swarm with reasonable limits
    handoff_swarm = Swarm(nodes=[handoff_agent, completion_agent], max_handoffs=10, max_iterations=10)
    handoff_agent._swarm_ref = handoff_swarm
    completion_agent._swarm_ref = handoff_swarm

    # Execute swarm - this should hand off from first agent to second agent
    result = handoff_swarm("Test handoff during execution")

    # Verify the handoff occurred
    assert result.status == Status.COMPLETED
    assert result.execution_count == 2  # Both agents should have executed
    assert len(result.node_history) == 2

    # Verify the handoff agent executed first
    assert result.node_history[0].node_id == "handoff_agent"

    # Verify the completion agent executed after handoff
    assert result.node_history[1].node_id == "completion_agent"

    # Verify both agents were called
    handoff_agent.invoke_async.assert_called()
    completion_agent.invoke_async.assert_called()

    # Test handoff when task is already completed
    completed_swarm = Swarm(nodes=[handoff_agent, completion_agent])
    completed_swarm.state.completion_status = Status.COMPLETED
    completed_swarm._handle_handoff(completed_swarm.nodes["completion_agent"], "test message", {"key": "value"})
    # Should not change current node when already completed


def test_swarm_tool_creation_and_execution():
    """Test swarm tool creation and execution with error handling."""
    error_agent = create_mock_agent("error_agent")
    error_swarm = Swarm(nodes=[error_agent])

    # Test tool execution with errors
    handoff_tool = error_swarm._create_handoff_tool()
    error_result = handoff_tool("nonexistent_agent", "test message")
    assert error_result["status"] == "error"
    assert "not found" in error_result["content"][0]["text"]


def test_swarm_failure_handling(mock_strands_tracer, mock_use_span):
    """Test swarm execution with agent failures."""
    # Test execution with agent failures
    failing_agent = create_mock_agent("failing_agent")
    failing_agent._should_fail = True  # Set failure flag after creation
    failing_swarm = Swarm(nodes=[failing_agent], node_timeout=1.0)

    # The swarm catches exceptions internally and sets status to FAILED
    result = failing_swarm("Test failure handling")
    assert result.status == Status.FAILED
    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


def test_swarm_metrics_handling():
    """Test swarm metrics handling with missing metrics."""
    no_metrics_agent = create_mock_agent("no_metrics", metrics=None)
    no_metrics_swarm = Swarm(nodes=[no_metrics_agent])

    result = no_metrics_swarm("Test no metrics")
    assert result.status == Status.COMPLETED


def test_swarm_auto_completion_without_handoff():
    """Test swarm auto-completion when no handoff occurs."""
    # Create a simple agent that doesn't hand off
    no_handoff_agent = create_mock_agent("no_handoff_agent", "Task completed without handoff")

    # Create a swarm with just this agent
    auto_complete_swarm = Swarm(nodes=[no_handoff_agent])

    # Execute swarm - this should complete automatically since there's no handoff
    result = auto_complete_swarm("Test auto-completion without handoff")

    # Verify the swarm completed successfully
    assert result.status == Status.COMPLETED
    assert result.execution_count == 1
    assert len(result.node_history) == 1
    assert result.node_history[0].node_id == "no_handoff_agent"

    # Verify the agent was called
    no_handoff_agent.invoke_async.assert_called()


def test_swarm_configurable_entry_point():
    """Test swarm with configurable entry point."""
    # Create multiple agents
    agent1 = create_mock_agent("agent1", "Agent 1 response")
    agent2 = create_mock_agent("agent2", "Agent 2 response")
    agent3 = create_mock_agent("agent3", "Agent 3 response")

    # Create swarm with agent2 as entry point
    swarm = Swarm([agent1, agent2, agent3], entry_point=agent2)

    # Verify entry point is set correctly
    assert swarm.entry_point is agent2

    # Execute swarm
    result = swarm("Test task")

    # Verify agent2 was the first to execute
    assert result.status == Status.COMPLETED
    assert len(result.node_history) == 1
    assert result.node_history[0].node_id == "agent2"


def test_swarm_invalid_entry_point():
    """Test swarm with invalid entry point raises error."""
    agent1 = create_mock_agent("agent1", "Agent 1 response")
    agent2 = create_mock_agent("agent2", "Agent 2 response")
    agent3 = create_mock_agent("agent3", "Agent 3 response")  # Not in swarm

    # Try to create swarm with agent not in the swarm
    with pytest.raises(ValueError, match="Entry point agent not found in swarm nodes"):
        Swarm([agent1, agent2], entry_point=agent3)


def test_swarm_default_entry_point():
    """Test swarm uses first agent as default entry point."""
    agent1 = create_mock_agent("agent1", "Agent 1 response")
    agent2 = create_mock_agent("agent2", "Agent 2 response")

    # Create swarm without specifying entry point
    swarm = Swarm([agent1, agent2])

    # Verify no explicit entry point is set
    assert swarm.entry_point is None

    # Execute swarm
    result = swarm("Test task")

    # Verify first agent was used as entry point
    assert result.status == Status.COMPLETED
    assert len(result.node_history) == 1
    assert result.node_history[0].node_id == "agent1"


def test_swarm_duplicate_agent_names():
    """Test swarm rejects agents with duplicate names."""
    agent1 = create_mock_agent("duplicate_name", "Agent 1 response")
    agent2 = create_mock_agent("duplicate_name", "Agent 2 response")

    # Try to create swarm with duplicate names
    with pytest.raises(ValueError, match="Node ID 'duplicate_name' is not unique"):
        Swarm([agent1, agent2])


def test_swarm_entry_point_same_name_different_object():
    """Test entry point validation with same name but different object."""
    agent1 = create_mock_agent("agent1", "Agent 1 response")
    agent2 = create_mock_agent("agent2", "Agent 2 response")

    # Create a different agent with same name as agent1
    different_agent_same_name = create_mock_agent("agent1", "Different agent response")

    # Try to use the different agent as entry point
    with pytest.raises(ValueError, match="Entry point agent not found in swarm nodes"):
        Swarm([agent1, agent2], entry_point=different_agent_same_name)


def test_swarm_validate_unsupported_features():
    """Test Swarm validation for session persistence and callbacks."""
    # Test with normal agent (should work)
    normal_agent = create_mock_agent("normal_agent")
    normal_agent._session_manager = None
    normal_agent.hooks = HookRegistry()

    swarm = Swarm([normal_agent])
    assert len(swarm.nodes) == 1

    # Test with session manager (should fail)
    mock_session_manager = Mock(spec=SessionManager)
    agent_with_session = create_mock_agent("agent_with_session")
    agent_with_session._session_manager = mock_session_manager
    agent_with_session.hooks = HookRegistry()

    with pytest.raises(ValueError, match="Session persistence is not supported for Swarm agents yet"):
        Swarm([agent_with_session])


@pytest.mark.asyncio
async def test_swarm_kwargs_passing(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying agents."""
    kwargs_agent = create_mock_agent("kwargs_agent", "Response with kwargs")
    kwargs_agent.invoke_async = Mock(side_effect=kwargs_agent.invoke_async)

    swarm = Swarm(nodes=[kwargs_agent])

    test_kwargs = {"custom_param": "test_value", "another_param": 42}
    result = await swarm.invoke_async("Test kwargs passing", test_kwargs)

    assert kwargs_agent.invoke_async.call_args.kwargs == test_kwargs
    assert result.status == Status.COMPLETED


def test_swarm_kwargs_passing_sync(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying agents in sync execution."""
    kwargs_agent = create_mock_agent("kwargs_agent", "Response with kwargs")
    kwargs_agent.invoke_async = Mock(side_effect=kwargs_agent.invoke_async)

    swarm = Swarm(nodes=[kwargs_agent])

    test_kwargs = {"custom_param": "test_value", "another_param": 42}
    result = swarm("Test kwargs passing sync", test_kwargs)

    assert kwargs_agent.invoke_async.call_args.kwargs == test_kwargs
    assert result.status == Status.COMPLETED
