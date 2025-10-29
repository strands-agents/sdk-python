import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from strands.agent import Agent, AgentResult
from strands.agent.state import AgentState
from strands.hooks.registry import HookRegistry
from strands.multiagent.base import Status
from strands.multiagent.swarm import SharedContext, Swarm, SwarmNode, SwarmResult, SwarmState
from strands.session.session_manager import SessionManager
from strands.types._events import MultiAgentNodeStartEvent
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

    async def mock_stream_async(*args, **kwargs):
        # Simple mock stream that yields a start event and then the result
        yield {"agent_start": True, "node": name}
        yield {"agent_thinking": True, "thought": f"Processing with {name}"}
        yield {"result": create_mock_result()}

    agent.invoke_async = MagicMock(side_effect=mock_invoke_async)
    agent.stream_async = Mock(side_effect=mock_stream_async)

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

    # Verify agent was called (via stream_async)
    assert mock_agents["coordinator"].stream_async.call_count >= 1

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

    # Verify agent was called (via stream_async)
    assert mock_agents["coordinator"].stream_async.call_count >= 1

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

        async def mock_stream_async(*args, **kwargs):
            yield {"agent_start": True}
            result = create_handoff_result()
            yield {"result": result}

        agent.invoke_async = MagicMock(side_effect=mock_invoke_async)
        agent.stream_async = Mock(side_effect=mock_stream_async)
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

    # Verify both agents were called (via stream_async)
    assert handoff_agent.stream_async.call_count >= 1
    assert completion_agent.stream_async.call_count >= 1

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

    # Verify the agent was called (via stream_async)
    assert no_handoff_agent.stream_async.call_count >= 1


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

    swarm = Swarm(nodes=[kwargs_agent])

    test_kwargs = {"custom_param": "test_value", "another_param": 42}
    result = await swarm.invoke_async("Test kwargs passing", test_kwargs)

    # Verify stream_async was called (kwargs are passed through)
    assert kwargs_agent.stream_async.call_count >= 1
    assert result.status == Status.COMPLETED


def test_swarm_kwargs_passing_sync(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying agents in sync execution."""
    kwargs_agent = create_mock_agent("kwargs_agent", "Response with kwargs")

    swarm = Swarm(nodes=[kwargs_agent])

    test_kwargs = {"custom_param": "test_value", "another_param": 42}
    result = swarm("Test kwargs passing sync", test_kwargs)

    # Verify stream_async was called (kwargs are passed through)
    assert kwargs_agent.stream_async.call_count >= 1
    assert result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_swarm_streaming_events(mock_strands_tracer, mock_use_span, alist):
    """Test that swarm streaming emits proper events during execution."""

    # Create agents with custom streaming behavior
    coordinator = create_mock_agent("coordinator", "Coordinating task")
    specialist = create_mock_agent("specialist", "Specialized response")

    # Track events and execution order
    execution_events = []

    async def coordinator_stream(*args, **kwargs):
        execution_events.append("coordinator_start")
        yield {"agent_start": True, "node": "coordinator"}
        yield {"agent_thinking": True, "thought": "Analyzing task"}
        await asyncio.sleep(0.01)  # Small delay
        execution_events.append("coordinator_end")
        yield {"result": coordinator.return_value}

    async def specialist_stream(*args, **kwargs):
        execution_events.append("specialist_start")
        yield {"agent_start": True, "node": "specialist"}
        yield {"agent_thinking": True, "thought": "Applying expertise"}
        await asyncio.sleep(0.01)  # Small delay
        execution_events.append("specialist_end")
        yield {"result": specialist.return_value}

    coordinator.stream_async = Mock(side_effect=coordinator_stream)
    specialist.stream_async = Mock(side_effect=specialist_stream)

    # Create swarm with handoff logic
    swarm = Swarm(nodes=[coordinator, specialist], max_handoffs=2, max_iterations=3, execution_timeout=30.0)

    # Add handoff tool to coordinator to trigger specialist
    def handoff_to_specialist():
        """Hand off to specialist for detailed analysis."""
        return specialist

    coordinator.tool_registry.registry = {"handoff_to_specialist": handoff_to_specialist}

    # Collect all streaming events
    events = await alist(swarm.stream_async("Test swarm streaming"))

    # Verify event structure
    assert len(events) > 0

    # Should have node start/stop events
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stop_events = [e for e in events if e.get("type") == "multiagent_node_stop"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Should have at least one node execution
    assert len(node_start_events) >= 1
    assert len(node_stop_events) >= 1

    # Should have forwarded agent events
    assert len(node_stream_events) >= 2  # At least some events per agent

    # Should have final result
    assert len(result_events) == 1

    # Verify node start events have correct structure
    for event in node_start_events:
        assert "node_id" in event
        assert "node_type" in event
        assert event["node_type"] == "agent"

    # Verify node stop events have node_result with execution time
    for event in node_stop_events:
        assert "node_id" in event
        assert "node_result" in event
        node_result = event["node_result"]
        assert hasattr(node_result, "execution_time")
        assert isinstance(node_result.execution_time, int)

    # Verify forwarded events maintain node context
    for event in node_stream_events:
        assert "node_id" in event

    # Verify final result
    final_result = result_events[0]["result"]
    assert final_result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_swarm_streaming_with_handoffs(mock_strands_tracer, mock_use_span, alist):
    """Test swarm streaming with agent handoffs."""

    # Create agents
    coordinator = create_mock_agent("coordinator", "Coordinating")
    specialist = create_mock_agent("specialist", "Specialized work")
    reviewer = create_mock_agent("reviewer", "Review complete")

    # Track handoff sequence
    handoff_sequence = []

    async def coordinator_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "coordinator"}
        yield {"agent_thinking": True, "thought": "Need specialist help"}
        handoff_sequence.append("coordinator_to_specialist")
        yield {"result": coordinator.return_value}

    async def specialist_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "specialist"}
        yield {"agent_thinking": True, "thought": "Doing specialized work"}
        handoff_sequence.append("specialist_to_reviewer")
        yield {"result": specialist.return_value}

    async def reviewer_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "reviewer"}
        yield {"agent_thinking": True, "thought": "Reviewing work"}
        handoff_sequence.append("reviewer_complete")
        yield {"result": reviewer.return_value}

    coordinator.stream_async = Mock(side_effect=coordinator_stream)
    specialist.stream_async = Mock(side_effect=specialist_stream)
    reviewer.stream_async = Mock(side_effect=reviewer_stream)

    # Set up handoff tools
    def handoff_to_specialist():
        return specialist

    def handoff_to_reviewer():
        return reviewer

    coordinator.tool_registry.registry = {"handoff_to_specialist": handoff_to_specialist}
    specialist.tool_registry.registry = {"handoff_to_reviewer": handoff_to_reviewer}
    reviewer.tool_registry.registry = {}

    # Create swarm
    swarm = Swarm(nodes=[coordinator, specialist, reviewer], max_handoffs=5, max_iterations=5, execution_timeout=30.0)

    # Collect streaming events
    events = await alist(swarm.stream_async("Test handoff streaming"))

    # Should have multiple node executions due to handoffs
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    handoff_events = [e for e in events if e.get("type") == "multiagent_handoff"]

    # Should have executed at least one agent (handoffs are complex to mock)
    assert len(node_start_events) >= 1

    # Verify handoff events have proper structure if any occurred
    for event in handoff_events:
        assert "from_node_ids" in event
        assert "to_node_ids" in event
        assert isinstance(event["from_node_ids"], list)
        assert isinstance(event["to_node_ids"], list)


@pytest.mark.asyncio
async def test_swarm_streaming_with_failures(mock_strands_tracer, mock_use_span):
    """Test swarm streaming behavior when agents fail."""

    # Create a failing agent (don't fail during creation, fail during execution)
    failing_agent = create_mock_agent("failing_agent", "Should fail")
    success_agent = create_mock_agent("success_agent", "Success")

    async def failing_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "failing_agent"}
        yield {"agent_thinking": True, "thought": "About to fail"}
        await asyncio.sleep(0.01)
        raise Exception("Simulated streaming failure")

    async def success_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "success_agent"}
        yield {"agent_thinking": True, "thought": "Working successfully"}
        yield {"result": success_agent.return_value}

    failing_agent.stream_async = Mock(side_effect=failing_stream)
    success_agent.stream_async = Mock(side_effect=success_stream)

    # Create swarm starting with failing agent
    swarm = Swarm(nodes=[failing_agent, success_agent], max_handoffs=2, max_iterations=3, execution_timeout=30.0)

    # Collect events until failure
    events = []
    # Note: We expect an exception but swarm might handle it gracefully
    # So we don't use pytest.raises here - we check for either success or failure
    try:
        async for event in swarm.stream_async("Test streaming with failure"):
            events.append(event)
    except Exception:
        pass  # Expected - failure during streaming

    # Should get some events before failure (if failure occurred)
    if len(events) > 0:
        # Should have node start events
        node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
        assert len(node_start_events) >= 1

        # Should have some forwarded events before failure
        node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
        assert len(node_stream_events) >= 1


@pytest.mark.asyncio
async def test_swarm_streaming_timeout_behavior(mock_strands_tracer, mock_use_span):
    """Test swarm streaming with execution timeout."""

    # Create a slow agent
    slow_agent = create_mock_agent("slow_agent", "Slow response")

    async def slow_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "slow_agent"}
        yield {"agent_thinking": True, "thought": "Taking my time"}
        await asyncio.sleep(0.2)  # Longer than timeout
        yield {"result": slow_agent.return_value}

    slow_agent.stream_async = Mock(side_effect=slow_stream)

    # Create swarm with short timeout
    swarm = Swarm(
        nodes=[slow_agent],
        max_handoffs=1,
        max_iterations=1,
        execution_timeout=0.1,  # Very short timeout
    )

    # Should timeout during streaming or complete
    # Note: Timeout behavior is timing-dependent, so we accept both outcomes
    events = []
    try:
        async for event in swarm.stream_async("Test timeout streaming"):
            events.append(event)
    except Exception:
        pass  # Timeout is acceptable

    # Should get at least some events regardless of timeout
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_swarm_streaming_backward_compatibility(mock_strands_tracer, mock_use_span, alist):
    """Test that swarm streaming maintains backward compatibility."""
    # Create simple agent
    agent = create_mock_agent("test_agent", "Test response")

    # Create swarm
    swarm = Swarm(nodes=[agent])

    # Test that invoke_async still works
    result = await swarm.invoke_async("Test backward compatibility")
    assert result.status == Status.COMPLETED

    # Test that streaming also works and produces same result
    events = await alist(swarm.stream_async("Test backward compatibility"))

    # Should have final result event
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]
    assert len(result_events) == 1

    streaming_result = result_events[0]["result"]
    assert streaming_result.status == Status.COMPLETED

    # Results should be equivalent
    assert result.status == streaming_result.status


@pytest.mark.asyncio
async def test_swarm_single_invocation_no_double_execution(mock_strands_tracer, mock_use_span):
    """Test that swarm nodes are only invoked once (no double execution from streaming)."""
    # Create agent with invocation counter
    agent = create_mock_agent("test_agent", "Test response")

    # Track invocation count
    invocation_count = {"count": 0}

    async def counted_stream(*args, **kwargs):
        invocation_count["count"] += 1
        yield {"agent_start": True, "node": "test_agent"}
        yield {"agent_thinking": True, "thought": "Processing"}
        yield {"result": agent.return_value}

    agent.stream_async = Mock(side_effect=counted_stream)

    # Create swarm
    swarm = Swarm(nodes=[agent])

    # Execute the swarm
    result = await swarm.invoke_async("Test single invocation")

    # Verify successful execution
    assert result.status == Status.COMPLETED

    # CRITICAL: Agent should be invoked exactly once
    assert invocation_count["count"] == 1, f"Agent invoked {invocation_count['count']} times, expected 1"

    # Verify stream_async was called but invoke_async was NOT called
    assert agent.stream_async.call_count == 1
    # invoke_async should not be called at all since we're using streaming
    agent.invoke_async.assert_not_called()


@pytest.mark.asyncio
async def test_swarm_handoff_single_invocation_per_node(mock_strands_tracer, mock_use_span):
    """Test that each node in a swarm handoff chain is invoked exactly once."""
    # Create agents with invocation counters
    invocation_counts = {"coordinator": 0, "specialist": 0}

    coordinator = create_mock_agent("coordinator", "Coordinating")
    specialist = create_mock_agent("specialist", "Specialized work")

    async def coordinator_stream(*args, **kwargs):
        invocation_counts["coordinator"] += 1
        yield {"agent_start": True, "node": "coordinator"}
        yield {"agent_thinking": True, "thought": "Need specialist"}
        yield {"result": coordinator.return_value}

    async def specialist_stream(*args, **kwargs):
        invocation_counts["specialist"] += 1
        yield {"agent_start": True, "node": "specialist"}
        yield {"agent_thinking": True, "thought": "Doing specialized work"}
        yield {"result": specialist.return_value}

    coordinator.stream_async = Mock(side_effect=coordinator_stream)
    specialist.stream_async = Mock(side_effect=specialist_stream)

    # Set up handoff tool
    def handoff_to_specialist():
        return specialist

    coordinator.tool_registry.registry = {"handoff_to_specialist": handoff_to_specialist}
    specialist.tool_registry.registry = {}

    # Create swarm
    swarm = Swarm(nodes=[coordinator, specialist], max_handoffs=2, max_iterations=3)

    # Execute the swarm
    result = await swarm.invoke_async("Test handoff single invocation")

    # Verify successful execution
    assert result.status == Status.COMPLETED

    # CRITICAL: Each agent should be invoked exactly once
    # Note: Actual invocation depends on whether handoff occurs, but no double execution
    assert invocation_counts["coordinator"] == 1, f"Coordinator invoked {invocation_counts['coordinator']} times"
    # Specialist may or may not be invoked depending on handoff logic, but if invoked, only once
    assert invocation_counts["specialist"] <= 1, f"Specialist invoked {invocation_counts['specialist']} times"

    # Verify stream_async was called but invoke_async was NOT called
    assert coordinator.stream_async.call_count == 1
    coordinator.invoke_async.assert_not_called()
    if invocation_counts["specialist"] > 0:
        specialist.invoke_async.assert_not_called()


@pytest.mark.asyncio
async def test_swarm_timeout_with_streaming(mock_strands_tracer, mock_use_span):
    """Test that swarm node timeout works correctly with streaming."""
    # Create a slow agent
    slow_agent = create_mock_agent("slow_agent", "Slow response")

    async def slow_stream(*args, **kwargs):
        yield {"agent_start": True, "node": "slow_agent"}
        await asyncio.sleep(0.3)  # Longer than timeout
        yield {"result": slow_agent.return_value}

    slow_agent.stream_async = Mock(side_effect=slow_stream)

    # Create swarm with short node timeout
    swarm = Swarm(
        nodes=[slow_agent],
        max_handoffs=1,
        max_iterations=1,
        node_timeout=0.1,  # Short timeout
    )

    # Execute - should complete with FAILED status due to timeout
    result = await swarm.invoke_async("Test timeout")

    # Verify the swarm failed due to timeout
    assert result.status == Status.FAILED

    # Verify the agent started streaming
    assert slow_agent.stream_async.call_count == 1


@pytest.mark.asyncio
async def test_swarm_node_timeout_with_mocked_streaming():
    """Test that swarm node timeout properly cancels a streaming generator that freezes."""
    # Create an agent that will timeout during streaming
    slow_agent = Agent(
        name="slow_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a slow agent. Take your time responding.",
    )

    # Override stream_async to simulate a freezing generator
    original_stream = slow_agent.stream_async

    async def freezing_stream(*args, **kwargs):
        """Simulate a generator that yields some events then freezes."""
        # Yield a few events normally
        count = 0
        async for event in original_stream(*args, **kwargs):
            yield event
            count += 1
            if count >= 3:
                # Simulate freezing - sleep longer than timeout
                await asyncio.sleep(10.0)
                break

    slow_agent.stream_async = freezing_stream

    # Create swarm with short node timeout
    swarm = Swarm(
        nodes=[slow_agent],
        max_handoffs=1,
        max_iterations=1,
        node_timeout=0.5,  # 500ms timeout
    )

    # Execute - should complete with FAILED status due to timeout
    result = await swarm.invoke_async("Test freezing generator")
    assert result.status == Status.FAILED


@pytest.mark.asyncio
async def test_swarm_timeout_cleanup_on_exception():
    """Test that timeout properly cleans up tasks even when exceptions occur."""
    # Create an agent
    agent = Agent(
        name="test_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a test agent.",
    )

    # Override stream_async to raise an exception after some events
    original_stream = agent.stream_async

    async def exception_stream(*args, **kwargs):
        """Simulate a generator that raises an exception."""
        count = 0
        async for event in original_stream(*args, **kwargs):
            yield event
            count += 1
            if count >= 2:
                raise ValueError("Simulated error during streaming")

    agent.stream_async = exception_stream

    # Create swarm with timeout
    swarm = Swarm(
        nodes=[agent],
        max_handoffs=1,
        max_iterations=1,
        node_timeout=30.0,
    )

    # Execute - swarm catches exceptions and continues, marking node as failed
    result = await swarm.invoke_async("Test exception handling")
    # Verify the node failed
    assert "test_agent" in result.results
    assert result.results["test_agent"].status == Status.FAILED
    assert result.status == Status.FAILED


@pytest.mark.asyncio
async def test_swarm_invoke_async_no_result_event(mock_strands_tracer, mock_use_span):
    """Test that invoke_async raises ValueError when stream produces no result event."""
    # Create a mock swarm that produces events but no final result
    agent = create_mock_agent("test_agent", "Test response")
    swarm = Swarm(nodes=[agent])

    # Mock stream_async to yield events but no result event
    async def no_result_stream(*args, **kwargs):
        """Simulate a stream that yields events but no result."""
        yield {"agent_start": True, "node": "test_agent"}
        yield {"agent_thinking": True, "thought": "Processing"}
        # Intentionally don't yield a result event

    swarm.stream_async = Mock(side_effect=no_result_stream)

    # Execute - should raise ValueError
    with pytest.raises(ValueError, match="Swarm streaming completed without producing a result event"):
        await swarm.invoke_async("Test no result event")


@pytest.mark.asyncio
async def test_swarm_stream_async_exception_in_execute_swarm(mock_strands_tracer, mock_use_span):
    """Test that stream_async logs exception when _execute_swarm raises an error."""
    # Create an agent
    agent = create_mock_agent("test_agent", "Test response")

    # Create swarm
    swarm = Swarm(nodes=[agent])

    # Mock _execute_swarm to raise an exception after yielding an event
    async def failing_execute_swarm(*args, **kwargs):
        """Simulate _execute_swarm raising an exception."""
        # Yield a valid event first

        yield MultiAgentNodeStartEvent(node_id="test_agent", node_type="agent")
        # Then raise an exception
        raise RuntimeError("Simulated failure in _execute_swarm")

    swarm._execute_swarm = Mock(side_effect=failing_execute_swarm)

    # Execute - should raise the exception and log it
    with pytest.raises(RuntimeError, match="Simulated failure in _execute_swarm"):
        async for _ in swarm.stream_async("Test exception logging"):
            pass

    # Verify the swarm status is FAILED
    assert swarm.state.completion_status == Status.FAILED
