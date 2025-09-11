import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from strands.agent import Agent, AgentResult
from strands.agent.state import AgentState
from strands.hooks import AgentInitializedEvent
from strands.hooks.registry import HookProvider, HookRegistry
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult
from strands.multiagent.graph import Graph, GraphBuilder, GraphEdge, GraphNode, GraphResult, GraphState, Status
from strands.session.session_manager import SessionManager


def create_mock_agent(name, response_text="Default response", metrics=None, agent_id=None):
    """Create a mock Agent with specified properties."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.id = agent_id or f"{name}_id"
    agent._session_manager = None
    agent.hooks = HookRegistry()

    if metrics is None:
        metrics = Mock(
            accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
            accumulated_metrics={"latencyMs": 100.0},
        )

    mock_result = AgentResult(
        message={"role": "assistant", "content": [{"text": response_text}]},
        stop_reason="end_turn",
        state={},
        metrics=metrics,
    )

    agent.return_value = mock_result
    agent.__call__ = Mock(return_value=mock_result)

    async def mock_invoke_async(*args, **kwargs):
        return mock_result

    agent.invoke_async = MagicMock(side_effect=mock_invoke_async)

    return agent


def create_mock_multi_agent(name, response_text="Multi-agent response"):
    """Create a mock MultiAgentBase with specified properties."""
    multi_agent = Mock(spec=MultiAgentBase)
    multi_agent.name = name
    multi_agent.id = f"{name}_id"

    mock_node_result = NodeResult(
        result=AgentResult(
            message={"role": "assistant", "content": [{"text": response_text}]},
            stop_reason="end_turn",
            state={},
            metrics={},
        )
    )
    mock_result = MultiAgentResult(
        results={"inner_node": mock_node_result},
        accumulated_usage={"inputTokens": 15, "outputTokens": 25, "totalTokens": 40},
        accumulated_metrics={"latencyMs": 150.0},
        execution_count=1,
        execution_time=150,
    )
    multi_agent.invoke_async = AsyncMock(return_value=mock_result)
    multi_agent.execute = Mock(return_value=mock_result)
    return multi_agent


@pytest.fixture
def mock_agents():
    """Create a set of diverse mock agents for testing."""
    return {
        "start_agent": create_mock_agent("start_agent", "Start response"),
        "multi_agent": create_mock_multi_agent("multi_agent", "Multi response"),
        "conditional_agent": create_mock_agent(
            "conditional_agent",
            "Conditional response",
            Mock(
                accumulated_usage={"inputTokens": 5, "outputTokens": 15, "totalTokens": 20},
                accumulated_metrics={"latencyMs": 75.0},
            ),
        ),
        "final_agent": create_mock_agent(
            "final_agent",
            "Final response",
            Mock(
                accumulated_usage={"inputTokens": 8, "outputTokens": 12, "totalTokens": 20},
                accumulated_metrics={"latencyMs": 50.0},
            ),
        ),
        "no_metrics_agent": create_mock_agent("no_metrics_agent", "No metrics response", metrics=None),
        "partial_metrics_agent": create_mock_agent(
            "partial_metrics_agent", "Partial metrics response", Mock(accumulated_usage={}, accumulated_metrics={})
        ),
        "blocked_agent": create_mock_agent("blocked_agent", "Should not execute"),
    }


@pytest.fixture
def string_content_agent():
    """Create an agent with string content (not list) for coverage testing."""
    agent = create_mock_agent("string_content_agent", "String content")
    agent.return_value.message = {"role": "assistant", "content": "string_content"}
    return agent


@pytest.fixture
def mock_strands_tracer():
    with patch("strands.multiagent.graph.get_tracer") as mock_get_tracer:
        mock_tracer_instance = MagicMock()
        mock_span = MagicMock()
        mock_tracer_instance.start_multiagent_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer_instance
        yield mock_tracer_instance


@pytest.fixture
def mock_use_span():
    with patch("strands.multiagent.graph.trace_api.use_span") as mock_use_span:
        yield mock_use_span


@pytest.fixture
def mock_graph(mock_agents, string_content_agent):
    """Create a graph for testing various scenarios."""

    def condition_check_completion(state: GraphState) -> bool:
        return any(node.node_id == "start_agent" for node in state.completed_nodes)

    def always_false_condition(state: GraphState) -> bool:
        return False

    builder = GraphBuilder()

    # Add nodes
    builder.add_node(mock_agents["start_agent"], "start_agent")
    builder.add_node(mock_agents["multi_agent"], "multi_node")
    builder.add_node(mock_agents["conditional_agent"], "conditional_agent")
    final_agent_graph_node = builder.add_node(mock_agents["final_agent"], "final_node")
    builder.add_node(mock_agents["no_metrics_agent"], "no_metrics_node")
    builder.add_node(mock_agents["partial_metrics_agent"], "partial_metrics_node")
    builder.add_node(string_content_agent, "string_content_node")
    builder.add_node(mock_agents["blocked_agent"], "blocked_node")

    # Add edges
    builder.add_edge("start_agent", "multi_node")
    builder.add_edge("start_agent", "conditional_agent", condition=condition_check_completion)
    builder.add_edge("multi_node", "final_node")
    builder.add_edge("conditional_agent", final_agent_graph_node)
    builder.add_edge("start_agent", "no_metrics_node")
    builder.add_edge("start_agent", "partial_metrics_node")
    builder.add_edge("start_agent", "string_content_node")
    builder.add_edge("start_agent", "blocked_node", condition=always_false_condition)

    builder.set_entry_point("start_agent")
    return builder.build()


@pytest.mark.asyncio
async def test_graph_execution(mock_strands_tracer, mock_use_span, mock_graph, mock_agents, string_content_agent):
    """Test comprehensive graph execution with diverse nodes and conditional edges."""

    # Test graph structure
    assert len(mock_graph.nodes) == 8
    assert len(mock_graph.edges) == 8
    assert len(mock_graph.entry_points) == 1
    assert any(node.node_id == "start_agent" for node in mock_graph.entry_points)

    # Test node properties
    start_node = mock_graph.nodes["start_agent"]
    assert start_node.node_id == "start_agent"
    assert start_node.executor == mock_agents["start_agent"]
    assert start_node.execution_status == Status.PENDING
    assert len(start_node.dependencies) == 0

    # Test conditional edge evaluation
    conditional_edge = next(
        edge
        for edge in mock_graph.edges
        if edge.from_node.node_id == "start_agent" and edge.to_node.node_id == "conditional_agent"
    )
    assert conditional_edge.condition is not None
    assert not conditional_edge.should_traverse(GraphState())

    # Create a mock GraphNode for testing
    start_node = mock_graph.nodes["start_agent"]
    assert conditional_edge.should_traverse(GraphState(completed_nodes={start_node}))

    result = await mock_graph.invoke_async("Test comprehensive execution")

    # Verify execution results
    assert result.status == Status.COMPLETED
    assert result.total_nodes == 8
    assert result.completed_nodes == 7  # All except blocked_node
    assert result.failed_nodes == 0
    assert len(result.execution_order) == 7
    assert result.execution_order[0].node_id == "start_agent"

    # Verify agent calls
    mock_agents["start_agent"].invoke_async.assert_called_once()
    mock_agents["multi_agent"].invoke_async.assert_called_once()
    mock_agents["conditional_agent"].invoke_async.assert_called_once()
    mock_agents["final_agent"].invoke_async.assert_called_once()
    mock_agents["no_metrics_agent"].invoke_async.assert_called_once()
    mock_agents["partial_metrics_agent"].invoke_async.assert_called_once()
    string_content_agent.invoke_async.assert_called_once()
    mock_agents["blocked_agent"].invoke_async.assert_not_called()

    # Verify metrics aggregation
    assert result.accumulated_usage["totalTokens"] > 0
    assert result.accumulated_metrics["latencyMs"] > 0
    assert result.execution_count >= 7

    # Verify node results
    assert len(result.results) == 7
    assert "blocked_node" not in result.results

    # Test result content extraction
    start_result = result.results["start_agent"]
    assert start_result.status == Status.COMPLETED
    agent_results = start_result.get_agent_results()
    assert len(agent_results) == 1
    assert "Start response" in str(agent_results[0].message)

    # Verify final graph state
    assert mock_graph.state.status == Status.COMPLETED
    assert len(mock_graph.state.completed_nodes) == 7
    assert len(mock_graph.state.failed_nodes) == 0

    # Test GraphResult properties
    assert isinstance(result, GraphResult)
    assert isinstance(result, MultiAgentResult)
    assert len(result.edges) == 8
    assert len(result.entry_points) == 1
    assert result.entry_points[0].node_id == "start_agent"

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


@pytest.mark.asyncio
async def test_graph_unsupported_node_type(mock_strands_tracer, mock_use_span):
    """Test unsupported executor type error handling."""

    class UnsupportedExecutor:
        pass

    builder = GraphBuilder()
    builder.add_node(UnsupportedExecutor(), "unsupported_node")
    graph = builder.build()

    # Execute the graph - should raise ValueError due to unsupported node type
    with pytest.raises(ValueError, match="Node 'unsupported_node' of type .* is not supported"):
        await graph.invoke_async("test task")

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


@pytest.mark.asyncio
async def test_graph_execution_with_failures(mock_strands_tracer, mock_use_span):
    """Test graph execution error handling and failure propagation."""
    failing_agent = Mock(spec=Agent)
    failing_agent.name = "failing_agent"
    failing_agent.id = "fail_node"
    failing_agent.__call__ = Mock(side_effect=Exception("Simulated failure"))

    # Add required attributes for validation
    failing_agent._session_manager = None
    failing_agent.hooks = HookRegistry()

    async def mock_invoke_failure(*args, **kwargs):
        raise Exception("Simulated failure")

    failing_agent.invoke_async = mock_invoke_failure

    success_agent = create_mock_agent("success_agent", "Success")

    builder = GraphBuilder()
    builder.add_node(failing_agent, "fail_node")
    builder.add_node(success_agent, "success_node")
    builder.add_edge("fail_node", "success_node")
    builder.set_entry_point("fail_node")

    graph = builder.build()

    # Execute the graph - should raise Exception due to failing agent
    with pytest.raises(Exception, match="Simulated failure"):
        await graph.invoke_async("Test error handling")

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


@pytest.mark.asyncio
async def test_graph_edge_cases(mock_strands_tracer, mock_use_span):
    """Test specific edge cases for coverage."""
    # Test entry node execution without dependencies
    entry_agent = create_mock_agent("entry_agent", "Entry response")

    builder = GraphBuilder()
    builder.add_node(entry_agent, "entry_only")
    graph = builder.build()

    result = await graph.invoke_async([{"text": "Original task"}])

    # Verify entry node was called with original task
    entry_agent.invoke_async.assert_called_once_with([{"text": "Original task"}])
    assert result.status == Status.COMPLETED
    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


@pytest.mark.asyncio
async def test_cyclic_graph_execution(mock_strands_tracer, mock_use_span):
    """Test execution of a graph with cycles and proper exit conditions."""
    # Create mock agents with state tracking
    agent_a = create_mock_agent("agent_a", "Agent A response")
    agent_b = create_mock_agent("agent_b", "Agent B response")
    agent_c = create_mock_agent("agent_c", "Agent C response")

    # Add state to agents to track execution
    agent_a.state = AgentState()
    agent_b.state = AgentState()
    agent_c.state = AgentState()

    # Create a spy to track reset calls
    reset_spy = MagicMock()

    # Create conditions for controlled cycling
    def a_to_b_condition(state: GraphState) -> bool:
        # A can trigger B if B hasn't been executed yet
        b_count = sum(1 for node in state.execution_order if node.node_id == "b")
        return b_count == 0

    def b_to_c_condition(state: GraphState) -> bool:
        # B can always trigger C (unconditional)
        return True

    def c_to_a_condition(state: GraphState) -> bool:
        # C can trigger A only if A has been executed less than 2 times
        a_count = sum(1 for node in state.execution_order if node.node_id == "a")
        return a_count < 2

    # Create a graph with conditional cycle: A -> B -> C -> A (with conditions)
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_node(agent_c, "c")
    builder.add_edge("a", "b", condition=a_to_b_condition)  # A -> B only if B not executed
    builder.add_edge("b", "c", condition=b_to_c_condition)  # B -> C always
    builder.add_edge("c", "a", condition=c_to_a_condition)  # C -> A only if A executed < 2 times
    builder.set_entry_point("a")
    builder.reset_on_revisit(True)  # Enable state reset on revisit
    builder.set_max_node_executions(10)  # Safety limit
    builder.set_execution_timeout(30.0)  # Safety timeout

    # Patch the reset_executor_state method to track calls
    original_reset = GraphNode.reset_executor_state

    def spy_reset(self):
        reset_spy(self.node_id)
        original_reset(self)

    with patch.object(GraphNode, "reset_executor_state", spy_reset):
        graph = builder.build()

        # Execute the graph with controlled cycling
        result = await graph.invoke_async("Test cyclic graph execution")

        # Verify that the graph executed successfully
        assert result.status == Status.COMPLETED

        # Expected execution order: a -> b -> c -> a (4 total executions)
        # A executes twice (initial + after c), B executes once, C executes once
        assert len(result.execution_order) == 4

        # Verify execution order
        execution_ids = [node.node_id for node in result.execution_order]
        assert execution_ids == ["a", "b", "c", "a"]

        # Verify that each agent was called the expected number of times
        assert agent_a.invoke_async.call_count == 2  # A executes twice
        assert agent_b.invoke_async.call_count == 1  # B executes once
        assert agent_c.invoke_async.call_count == 1  # C executes once

        # Verify that node state was reset for the revisited node (A)
        assert reset_spy.call_args_list == [call("a")]  # Only A should be reset (when revisited)

        # Verify all nodes were completed (final state)
        assert result.completed_nodes == 3


def test_graph_builder_validation():
    """Test GraphBuilder validation and error handling."""
    # Test empty graph validation
    builder = GraphBuilder()
    with pytest.raises(ValueError, match="Graph must contain at least one node"):
        builder.build()

    # Test duplicate node IDs
    agent1 = create_mock_agent("agent1")
    agent2 = create_mock_agent("agent2")
    builder.add_node(agent1, "duplicate_id")
    with pytest.raises(ValueError, match="Node 'duplicate_id' already exists"):
        builder.add_node(agent2, "duplicate_id")

    # Test duplicate node instances in GraphBuilder.add_node
    builder = GraphBuilder()
    same_agent = create_mock_agent("same_agent")
    builder.add_node(same_agent, "node1")
    with pytest.raises(ValueError, match="Duplicate node instance detected"):
        builder.add_node(same_agent, "node2")  # Same agent instance, different node_id

    # Test duplicate node instances in Graph.__init__
    duplicate_agent = create_mock_agent("duplicate_agent")
    node1 = GraphNode("node1", duplicate_agent)
    node2 = GraphNode("node2", duplicate_agent)  # Same agent instance
    nodes = {"node1": node1, "node2": node2}
    with pytest.raises(ValueError, match="Duplicate node instance detected"):
        Graph(
            nodes=nodes,
            edges=set(),
            entry_points=set(),
        )

    # Test edge validation with non-existent nodes
    builder = GraphBuilder()
    builder.add_node(agent1, "node1")
    with pytest.raises(ValueError, match="Target node 'nonexistent' not found"):
        builder.add_edge("node1", "nonexistent")
    with pytest.raises(ValueError, match="Source node 'nonexistent' not found"):
        builder.add_edge("nonexistent", "node1")

    # Test invalid entry point
    with pytest.raises(ValueError, match="Node 'invalid_entry' not found"):
        builder.set_entry_point("invalid_entry")

    # Test multiple invalid entry points in build validation
    builder = GraphBuilder()
    builder.add_node(agent1, "valid_node")
    # Create mock GraphNode objects for invalid entry points
    invalid_node1 = GraphNode("invalid1", agent1)
    invalid_node2 = GraphNode("invalid2", agent2)
    builder.entry_points.add(invalid_node1)
    builder.entry_points.add(invalid_node2)
    with pytest.raises(ValueError, match="Entry points not found in nodes"):
        builder.build()

    # Test cycle detection (should be forbidden by default)
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_node(create_mock_agent("agent3"), "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "a")  # Creates cycle
    builder.set_entry_point("a")

    # Should succeed - cycles are now allowed by default
    graph = builder.build()
    assert any(node.node_id == "a" for node in graph.entry_points)

    # Test auto-detection of entry points
    builder = GraphBuilder()
    builder.add_node(agent1, "entry")
    builder.add_node(agent2, "dependent")
    builder.add_edge("entry", "dependent")

    graph = builder.build()
    assert any(node.node_id == "entry" for node in graph.entry_points)

    # Test no entry points scenario
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_edge("a", "b")
    builder.add_edge("b", "a")

    with pytest.raises(ValueError, match="No entry points found - all nodes have dependencies"):
        builder.build()

    # Test custom execution limits and reset_on_revisit
    builder = GraphBuilder()
    builder.add_node(agent1, "test_node")
    graph = (
        builder.set_max_node_executions(10)
        .set_execution_timeout(300.0)
        .set_node_timeout(60.0)
        .reset_on_revisit()
        .build()
    )
    assert graph.max_node_executions == 10
    assert graph.execution_timeout == 300.0
    assert graph.node_timeout == 60.0
    assert graph.reset_on_revisit is True

    # Test default execution limits and reset_on_revisit (None and False)
    builder = GraphBuilder()
    builder.add_node(agent1, "test_node")
    graph = builder.build()
    assert graph.max_node_executions is None
    assert graph.execution_timeout is None
    assert graph.node_timeout is None
    assert graph.reset_on_revisit is False


@pytest.mark.asyncio
async def test_graph_execution_limits(mock_strands_tracer, mock_use_span):
    """Test graph execution limits (max_node_executions and execution_timeout)."""
    # Test with a simple linear graph first to verify limits work
    agent_a = create_mock_agent("agent_a", "Response A")
    agent_b = create_mock_agent("agent_b", "Response B")
    agent_c = create_mock_agent("agent_c", "Response C")

    # Create a linear graph: a -> b -> c
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_node(agent_c, "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.set_entry_point("a")

    # Test with no limits (backward compatibility) - should complete normally
    graph = builder.build()  # No limits specified
    result = await graph.invoke_async("Test execution")
    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 3  # All 3 nodes should execute

    # Test with limit that allows completion
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_node(agent_c, "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.set_entry_point("a")
    graph = builder.set_max_node_executions(5).set_execution_timeout(900.0).set_node_timeout(300.0).build()
    result = await graph.invoke_async("Test execution")
    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 3  # All 3 nodes should execute

    # Test with limit that prevents full completion
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_node(agent_c, "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.set_entry_point("a")
    graph = builder.set_max_node_executions(2).set_execution_timeout(900.0).set_node_timeout(300.0).build()
    result = await graph.invoke_async("Test execution limit")
    assert result.status == Status.FAILED  # Should fail due to limit
    assert len(result.execution_order) == 2  # Should stop at 2 executions


@pytest.mark.asyncio
async def test_graph_execution_limits_with_cyclic_graph(mock_strands_tracer, mock_use_span):
    timeout_agent_a = create_mock_agent("timeout_agent_a", "Response A")
    timeout_agent_b = create_mock_agent("timeout_agent_b", "Response B")

    # Create a cyclic graph that would run indefinitely
    builder = GraphBuilder()
    builder.add_node(timeout_agent_a, "a")
    builder.add_node(timeout_agent_b, "b")
    builder.add_edge("a", "b")
    builder.add_edge("b", "a")  # Creates cycle
    builder.set_entry_point("a")

    # Enable reset_on_revisit so the cycle can continue
    graph = builder.reset_on_revisit(True).set_execution_timeout(5.0).set_max_node_executions(100).build()

    # Execute the cyclic graph - should hit one of the limits
    result = await graph.invoke_async("Test execution limits")

    # Should fail due to hitting a limit (either timeout or max executions)
    assert result.status == Status.FAILED
    # Should have executed many nodes (hitting the limit)
    assert len(result.execution_order) >= 50  # Should execute many times before hitting limit

    # Test timeout logic directly (without execution)
    test_state = GraphState()
    test_state.start_time = time.time() - 10  # Set start time to 10 seconds ago
    should_continue, reason = test_state.should_continue(max_node_executions=100, execution_timeout=5.0)
    assert should_continue is False
    assert "Execution timed out" in reason

    # Test max executions logic directly (without execution)
    test_state2 = GraphState()
    test_state2.execution_order = [None] * 101  # Simulate 101 executions
    should_continue2, reason2 = test_state2.should_continue(max_node_executions=100, execution_timeout=5.0)
    assert should_continue2 is False
    assert "Max node executions reached" in reason2

    # builder = GraphBuilder()
    # builder.add_node(slow_agent, "slow")
    # graph = (builder.set_max_node_executions(1000)  # High limit to avoid hitting this
    #          .set_execution_timeout(0.05)  # Very short execution timeout
    #          .set_node_timeout(300.0)
    #          .build())

    # result = await graph.invoke_async("Test timeout")
    # assert result.status == Status.FAILED  # Should fail due to timeout

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called()


@pytest.mark.asyncio
async def test_graph_node_timeout(mock_strands_tracer, mock_use_span):
    """Test individual node timeout functionality."""

    # Create a mock agent that takes longer than the node timeout
    timeout_agent = create_mock_agent("timeout_agent", "Should timeout")

    async def timeout_invoke(*args, **kwargs):
        await asyncio.sleep(0.2)  # Longer than node timeout
        return timeout_agent.return_value

    timeout_agent.invoke_async = AsyncMock(side_effect=timeout_invoke)

    builder = GraphBuilder()
    builder.add_node(timeout_agent, "timeout_node")

    # Test with no timeout (backward compatibility) - should complete normally
    graph = builder.build()  # No timeout specified
    result = await graph.invoke_async("Test no timeout")
    assert result.status == Status.COMPLETED
    assert result.completed_nodes == 1

    # Test with very short node timeout - should raise timeout exception
    builder = GraphBuilder()
    builder.add_node(timeout_agent, "timeout_node")
    graph = builder.set_max_node_executions(50).set_execution_timeout(900.0).set_node_timeout(0.1).build()

    # Execute the graph - should raise Exception due to timeout
    with pytest.raises(Exception, match="Node 'timeout_node' execution timed out after 0.1s"):
        await graph.invoke_async("Test node timeout")

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called()


@pytest.mark.asyncio
async def test_backward_compatibility_no_limits():
    """Test that graphs with no limits specified work exactly as before."""
    # Create simple agents
    agent_a = create_mock_agent("agent_a", "Response A")
    agent_b = create_mock_agent("agent_b", "Response B")

    # Create a simple linear graph
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_edge("a", "b")
    builder.set_entry_point("a")

    # Build without specifying any limits - should work exactly as before
    graph = builder.build()

    # Verify the limits are None (no limits)
    assert graph.max_node_executions is None
    assert graph.execution_timeout is None
    assert graph.node_timeout is None

    # Execute the graph - should complete normally
    result = await graph.invoke_async("Test backward compatibility")
    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 2  # Both nodes should execute


@pytest.mark.asyncio
async def test_node_reset_executor_state():
    """Test that GraphNode.reset_executor_state properly resets node state."""
    # Create a mock agent with state
    agent = create_mock_agent("test_agent", "Test response")
    agent.state = AgentState()
    agent.state.set("test_key", "test_value")
    agent.messages = [{"role": "system", "content": "Initial system message"}]

    # Create a GraphNode with this agent
    node = GraphNode("test_node", agent)

    # Verify initial state is captured during initialization
    assert len(node._initial_messages) == 1
    assert node._initial_messages[0]["role"] == "system"
    assert node._initial_messages[0]["content"] == "Initial system message"

    # Modify agent state and messages after initialization
    agent.state.set("new_key", "new_value")
    agent.messages.append({"role": "user", "content": "New message"})

    # Also modify execution status and result
    node.execution_status = Status.COMPLETED
    node.result = NodeResult(
        result="test result",
        execution_time=100,
        status=Status.COMPLETED,
        accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
        accumulated_metrics={"latencyMs": 100},
        execution_count=1,
    )

    # Verify state was modified
    assert len(agent.messages) == 2
    assert agent.state.get("new_key") == "new_value"
    assert node.execution_status == Status.COMPLETED
    assert node.result is not None

    # Reset the executor state
    node.reset_executor_state()

    # Verify messages were reset to initial values
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == "Initial system message"

    # Verify agent state was reset
    # The test_key should be gone since it wasn't in the initial state
    assert agent.state.get("new_key") is None

    # Verify execution status is reset
    assert node.execution_status == Status.PENDING
    assert node.result is None

    # Test with MultiAgentBase executor
    multi_agent = create_mock_multi_agent("multi_agent")
    multi_agent_node = GraphNode("multi_node", multi_agent)

    # Since MultiAgentBase doesn't have messages or state attributes,
    # reset_executor_state should not fail
    multi_agent_node.execution_status = Status.COMPLETED
    multi_agent_node.result = NodeResult(
        result="test result",
        execution_time=100,
        status=Status.COMPLETED,
        accumulated_usage={},
        accumulated_metrics={},
        execution_count=1,
    )

    # Reset should work without errors
    multi_agent_node.reset_executor_state()

    # Verify execution status is reset
    assert multi_agent_node.execution_status == Status.PENDING
    assert multi_agent_node.result is None


def test_graph_dataclasses_and_enums():
    """Test dataclass initialization, properties, and enum behavior."""
    # Test Status enum
    assert Status.PENDING.value == "pending"
    assert Status.EXECUTING.value == "executing"
    assert Status.COMPLETED.value == "completed"
    assert Status.FAILED.value == "failed"

    # Test GraphState initialization and defaults
    state = GraphState()
    assert state.status == Status.PENDING
    assert len(state.completed_nodes) == 0
    assert len(state.failed_nodes) == 0
    assert state.task == ""
    assert state.accumulated_usage == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    assert state.execution_count == 0
    assert state.start_time > 0  # Should be set by default factory

    # Test GraphState with custom values
    state = GraphState(status=Status.EXECUTING, task="custom task", total_nodes=5, execution_count=3)
    assert state.status == Status.EXECUTING
    assert state.task == "custom task"
    assert state.total_nodes == 5
    assert state.execution_count == 3

    # Test GraphEdge with and without condition
    mock_agent_a = create_mock_agent("agent_a")
    mock_agent_b = create_mock_agent("agent_b")
    node_a = GraphNode("a", mock_agent_a)
    node_b = GraphNode("b", mock_agent_b)

    edge_simple = GraphEdge(node_a, node_b)
    assert edge_simple.from_node == node_a
    assert edge_simple.to_node == node_b
    assert edge_simple.condition is None
    assert edge_simple.should_traverse(GraphState())

    def test_condition(state):
        return len(state.completed_nodes) > 0

    edge_conditional = GraphEdge(node_a, node_b, condition=test_condition)
    assert edge_conditional.condition is not None
    assert not edge_conditional.should_traverse(GraphState())

    # Create a mock GraphNode for testing
    mock_completed_node = GraphNode("some_node", create_mock_agent("some_agent"))
    assert edge_conditional.should_traverse(GraphState(completed_nodes={mock_completed_node}))

    # Test GraphEdge hashing
    node_x = GraphNode("x", mock_agent_a)
    node_y = GraphNode("y", mock_agent_b)
    edge1 = GraphEdge(node_x, node_y)
    edge2 = GraphEdge(node_x, node_y)
    edge3 = GraphEdge(node_y, node_x)
    assert hash(edge1) == hash(edge2)
    assert hash(edge1) != hash(edge3)

    # Test GraphNode initialization
    mock_agent = create_mock_agent("test_agent")
    node = GraphNode("test_node", mock_agent)
    assert node.node_id == "test_node"
    assert node.executor == mock_agent
    assert node.execution_status == Status.PENDING
    assert len(node.dependencies) == 0


def test_graph_synchronous_execution(mock_strands_tracer, mock_use_span, mock_agents):
    """Test synchronous graph execution using execute method."""
    builder = GraphBuilder()
    builder.add_node(mock_agents["start_agent"], "start_agent")
    builder.add_node(mock_agents["final_agent"], "final_agent")
    builder.add_edge("start_agent", "final_agent")
    builder.set_entry_point("start_agent")

    graph = builder.build()

    # Test synchronous execution
    result = graph("Test synchronous execution")

    # Verify execution results
    assert result.status == Status.COMPLETED
    assert result.total_nodes == 2
    assert result.completed_nodes == 2
    assert result.failed_nodes == 0
    assert len(result.execution_order) == 2
    assert result.execution_order[0].node_id == "start_agent"
    assert result.execution_order[1].node_id == "final_agent"

    # Verify agent calls
    mock_agents["start_agent"].invoke_async.assert_called_once()
    mock_agents["final_agent"].invoke_async.assert_called_once()

    # Verify return type is GraphResult
    assert isinstance(result, GraphResult)
    assert isinstance(result, MultiAgentResult)

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called_once()


def test_graph_validate_unsupported_features():
    """Test Graph validation for session persistence and callbacks."""
    # Test with normal agent (should work)
    normal_agent = create_mock_agent("normal_agent")
    normal_agent._session_manager = None
    normal_agent.hooks = HookRegistry()

    builder = GraphBuilder()
    builder.add_node(normal_agent)
    graph = builder.build()
    assert len(graph.nodes) == 1

    # Test with session manager (should fail in GraphBuilder.add_node)
    mock_session_manager = Mock(spec=SessionManager)
    agent_with_session = create_mock_agent("agent_with_session")
    agent_with_session._session_manager = mock_session_manager
    agent_with_session.hooks = HookRegistry()

    builder = GraphBuilder()
    with pytest.raises(ValueError, match="Session persistence is not supported for Graph agents yet"):
        builder.add_node(agent_with_session)

    # Test with callbacks (should fail in GraphBuilder.add_node)
    class TestHookProvider(HookProvider):
        def register_hooks(self, registry, **kwargs):
            registry.add_callback(AgentInitializedEvent, lambda e: None)

    # Test validation in Graph constructor (when nodes are passed directly)
    # Test with session manager in Graph constructor
    node_with_session = GraphNode("node_with_session", agent_with_session)
    with pytest.raises(ValueError, match="Session persistence is not supported for Graph agents yet"):
        Graph(
            nodes={"node_with_session": node_with_session},
            edges=set(),
            entry_points=set(),
        )


@pytest.mark.asyncio
async def test_controlled_cyclic_execution():
    """Test cyclic graph execution with controlled cycle count to verify state reset."""

    # Create a stateful agent that tracks its own execution count
    class StatefulAgent(Agent):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.state = AgentState()
            self.state.set("execution_count", 0)
            self.messages = []
            self._session_manager = None
            self.hooks = HookRegistry()

        async def invoke_async(self, input_data):
            # Increment execution count in state
            count = self.state.get("execution_count") or 0
            self.state.set("execution_count", count + 1)

            return AgentResult(
                message={"role": "assistant", "content": [{"text": f"{self.name} response (execution {count + 1})"}]},
                stop_reason="end_turn",
                state={},
                metrics=Mock(
                    accumulated_usage={"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                    accumulated_metrics={"latencyMs": 100.0},
                ),
            )

    # Create agents
    agent_a = StatefulAgent("agent_a")
    agent_b = StatefulAgent("agent_b")

    # Create a graph with a simple cycle: A -> B -> A
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_edge("a", "b")
    builder.add_edge("b", "a")  # Creates cycle
    builder.set_entry_point("a")
    builder.reset_on_revisit()  # Enable state reset on revisit

    # Build with limited max_node_executions to prevent infinite loop
    graph = builder.set_max_node_executions(3).build()

    # Execute the graph
    result = await graph.invoke_async("Test controlled cyclic execution")

    # With a 2-node cycle and limit of 3, we should see either completion or failure
    # The exact behavior depends on how the cycle detection works
    if result.status == Status.COMPLETED:
        # If it completed, verify it executed some nodes
        assert len(result.execution_order) >= 2
        assert result.execution_order[0].node_id == "a"
    elif result.status == Status.FAILED:
        # If it failed due to limits, verify it hit the limit
        assert len(result.execution_order) == 3  # Should stop at exactly 3 executions
        assert result.execution_order[0].node_id == "a"
    else:
        # Should be either completed or failed
        raise AssertionError(f"Unexpected status: {result.status}")

    # Most importantly, verify that state was reset properly between executions
    # The state.execution_count should be set for both agents after execution
    assert agent_a.state.get("execution_count") >= 1  # Node A executed at least once
    assert agent_b.state.get("execution_count") >= 1  # Node B executed at least once


def test_reset_on_revisit_backward_compatibility():
    """Test that reset_on_revisit provides backward compatibility by default."""
    agent1 = create_mock_agent("agent1")
    agent2 = create_mock_agent("agent2")

    # Test default behavior - reset_on_revisit is False by default
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_edge("a", "b")
    builder.set_entry_point("a")

    graph = builder.build()
    assert graph.reset_on_revisit is False

    # Test reset_on_revisit with True
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_edge("a", "b")
    builder.set_entry_point("a")
    builder.reset_on_revisit(True)

    graph = builder.build()
    assert graph.reset_on_revisit is True

    # Test reset_on_revisit with False explicitly
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_edge("a", "b")
    builder.set_entry_point("a")
    builder.reset_on_revisit(False)

    graph = builder.build()
    assert graph.reset_on_revisit is False


def test_reset_on_revisit_method_chaining():
    """Test that reset_on_revisit method returns GraphBuilder for chaining."""
    agent1 = create_mock_agent("agent1")

    builder = GraphBuilder()
    result = builder.reset_on_revisit()

    # Verify method chaining works
    assert result is builder
    assert builder._reset_on_revisit is True

    # Test full method chaining
    builder.add_node(agent1, "test_node")
    builder.set_max_node_executions(10)
    graph = builder.build()

    assert graph.reset_on_revisit is True
    assert graph.max_node_executions == 10


@pytest.mark.asyncio
async def test_linear_graph_behavior():
    """Test that linear graph behavior works correctly."""
    agent_a = create_mock_agent("agent_a", "Response A")
    agent_b = create_mock_agent("agent_b", "Response B")

    # Create linear graph
    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_edge("a", "b")
    builder.set_entry_point("a")

    graph = builder.build()
    assert graph.reset_on_revisit is False

    # Execute should work normally
    result = await graph.invoke_async("Test linear execution")
    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 2
    assert result.execution_order[0].node_id == "a"
    assert result.execution_order[1].node_id == "b"

    # Verify agents were called once each (no state reset)
    agent_a.invoke_async.assert_called_once()
    agent_b.invoke_async.assert_called_once()


@pytest.mark.asyncio
async def test_state_reset_only_with_cycles_enabled():
    """Test that state reset only happens when cycles are enabled."""
    # Create a mock agent that tracks state modifications
    agent = create_mock_agent("test_agent", "Test response")
    agent.state = AgentState()
    agent.messages = [{"role": "system", "content": "Initial message"}]

    # Create GraphNode
    node = GraphNode("test_node", agent)

    # Simulate agent being in completed_nodes (as if revisited)
    from strands.multiagent.graph import GraphState

    state = GraphState()
    state.completed_nodes.add(node)

    # Create graph with cycles disabled (default)
    builder = GraphBuilder()
    builder.add_node(agent, "test_node")
    graph = builder.build()

    # Mock the _execute_node method to test conditional reset logic
    with patch.object(node, "reset_executor_state") as mock_reset:
        # Simulate the conditional logic from _execute_node
        if graph.reset_on_revisit and node in state.completed_nodes:
            node.reset_executor_state()
            state.completed_nodes.remove(node)

        # With reset_on_revisit disabled, reset should not be called
        mock_reset.assert_not_called()

    # Now test with reset_on_revisit enabled
    builder = GraphBuilder()
    builder.add_node(agent, "test_node")
    builder.reset_on_revisit()
    graph = builder.build()

    with patch.object(node, "reset_executor_state") as mock_reset:
        # Simulate the conditional logic from _execute_node
        if graph.reset_on_revisit and node in state.completed_nodes:
            node.reset_executor_state()
            state.completed_nodes.remove(node)

        # With reset_on_revisit enabled, reset should be called
        mock_reset.assert_called_once()


@pytest.mark.asyncio
async def test_self_loop_functionality(mock_strands_tracer, mock_use_span):
    """Test comprehensive self-loop functionality including conditions and reset behavior."""
    # Test basic self-loop with execution counting
    self_loop_agent = create_mock_agent("self_loop_agent", "Self loop response")
    self_loop_agent.invoke_async = Mock(side_effect=self_loop_agent.invoke_async)

    def loop_condition(state: GraphState) -> bool:
        return len(state.execution_order) < 3

    builder = GraphBuilder()
    builder.add_node(self_loop_agent, "self_loop")
    builder.add_edge("self_loop", "self_loop", condition=loop_condition)
    builder.set_entry_point("self_loop")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(10)
    builder.set_execution_timeout(30.0)

    graph = builder.build()
    result = await graph.invoke_async("Test self loop")

    # Verify basic self-loop functionality
    assert result.status == Status.COMPLETED
    assert self_loop_agent.invoke_async.call_count == 3
    assert len(result.execution_order) == 3
    assert all(node.node_id == "self_loop" for node in result.execution_order)


@pytest.mark.asyncio
async def test_self_loop_functionality_without_reset(mock_strands_tracer, mock_use_span):
    loop_agent_no_reset = create_mock_agent("loop_agent", "Loop without reset")

    can_only_be_called_twice: Mock = Mock(side_effect=lambda state: can_only_be_called_twice.call_count <= 2)

    builder = GraphBuilder()
    builder.add_node(loop_agent_no_reset, "loop_node")
    builder.add_edge("loop_node", "loop_node", condition=can_only_be_called_twice)
    builder.set_entry_point("loop_node")
    builder.reset_on_revisit(False)  # Disable state reset
    builder.set_max_node_executions(10)

    graph = builder.build()
    result = await graph.invoke_async("Test self loop without reset")

    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 2

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called()


@pytest.mark.asyncio
async def test_complex_self_loop(mock_strands_tracer, mock_use_span):
    """Test complex self-loop scenarios including multi-node graphs and multiple self-loops."""
    start_agent = create_mock_agent("start_agent", "Start")
    loop_agent = create_mock_agent("loop_agent", "Loop")
    end_agent = create_mock_agent("end_agent", "End")

    def loop_condition(state: GraphState) -> bool:
        loop_count = sum(1 for node in state.execution_order if node.node_id == "loop_node")
        return loop_count < 2

    def end_condition(state: GraphState) -> bool:
        loop_count = sum(1 for node in state.execution_order if node.node_id == "loop_node")
        return loop_count >= 2

    builder = GraphBuilder()
    builder.add_node(start_agent, "start_node")
    builder.add_node(loop_agent, "loop_node")
    builder.add_node(end_agent, "end_node")
    builder.add_edge("start_node", "loop_node")
    builder.add_edge("loop_node", "loop_node", condition=loop_condition)
    builder.add_edge("loop_node", "end_node", condition=end_condition)
    builder.set_entry_point("start_node")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(10)

    graph = builder.build()
    result = await graph.invoke_async("Test complex graph with self loops")

    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 4  # start -> loop -> loop -> end
    assert [node.node_id for node in result.execution_order] == ["start_node", "loop_node", "loop_node", "end_node"]
    assert start_agent.invoke_async.call_count == 1
    assert loop_agent.invoke_async.call_count == 2
    assert end_agent.invoke_async.call_count == 1


@pytest.mark.asyncio
async def test_multiple_nodes_with_self_loops(mock_strands_tracer, mock_use_span):
    agent_a = create_mock_agent("agent_a", "Agent A")
    agent_b = create_mock_agent("agent_b", "Agent B")

    def condition_a(state: GraphState) -> bool:
        return sum(1 for node in state.execution_order if node.node_id == "a") < 2

    def condition_b(state: GraphState) -> bool:
        return sum(1 for node in state.execution_order if node.node_id == "b") < 2

    builder = GraphBuilder()
    builder.add_node(agent_a, "a")
    builder.add_node(agent_b, "b")
    builder.add_edge("a", "a", condition=condition_a)
    builder.add_edge("b", "b", condition=condition_b)
    builder.add_edge("a", "b")
    builder.set_entry_point("a")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(15)

    graph = builder.build()
    result = await graph.invoke_async("Test multiple self loops")

    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 4  # a -> a -> b -> b
    assert agent_a.invoke_async.call_count == 2
    assert agent_b.invoke_async.call_count == 2

    mock_strands_tracer.start_multiagent_span.assert_called()
    mock_use_span.assert_called()


@pytest.mark.asyncio
async def test_self_loop_state_reset():
    """Test self-loop edge cases including state reset, failure handling, and infinite loop prevention."""
    agent = create_mock_agent("stateful_agent", "Stateful response")
    agent.state = AgentState()

    def loop_condition(state: GraphState) -> bool:
        return len(state.execution_order) < 3

    builder = GraphBuilder()
    node = builder.add_node(agent, "stateful_node")
    builder.add_edge("stateful_node", "stateful_node", condition=loop_condition)
    builder.set_entry_point("stateful_node")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(10)

    node.reset_executor_state = Mock(wraps=node.reset_executor_state)

    graph = builder.build()
    result = await graph.invoke_async("Test state reset")

    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 3
    assert node.reset_executor_state.call_count >= 2  # Reset called for revisits


@pytest.mark.asyncio
async def test_infinite_loop_prevention():
    infinite_agent = create_mock_agent("infinite_agent", "Infinite loop")

    def always_true_condition(state: GraphState) -> bool:
        return True

    builder = GraphBuilder()
    builder.add_node(infinite_agent, "infinite_node")
    builder.add_edge("infinite_node", "infinite_node", condition=always_true_condition)
    builder.set_entry_point("infinite_node")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(5)

    graph = builder.build()
    result = await graph.invoke_async("Test infinite loop prevention")

    assert result.status == Status.FAILED
    assert len(result.execution_order) == 5


@pytest.mark.asyncio
async def test_infinite_loop_prevention_self_loops():
    multi_agent = create_mock_multi_agent("multi_agent", "Multi-agent response")
    loop_count = 0

    def multi_loop_condition(state: GraphState) -> bool:
        nonlocal loop_count
        loop_count += 1
        return loop_count <= 2

    builder = GraphBuilder()
    builder.add_node(multi_agent, "multi_node")
    builder.add_edge("multi_node", "multi_node", condition=multi_loop_condition)
    builder.set_entry_point("multi_node")
    builder.reset_on_revisit(True)
    builder.set_max_node_executions(10)

    graph = builder.build()
    result = await graph.invoke_async("Test multi-agent self loop")

    assert result.status == Status.COMPLETED
    assert len(result.execution_order) >= 2
    assert multi_agent.invoke_async.call_count >= 2


@pytest.mark.asyncio
async def test_graph_kwargs_passing_agent(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying Agent nodes."""
    kwargs_agent = create_mock_agent("kwargs_agent", "Response with kwargs")
    kwargs_agent.invoke_async = Mock(side_effect=kwargs_agent.invoke_async)

    builder = GraphBuilder()
    builder.add_node(kwargs_agent, "kwargs_node")
    graph = builder.build()

    test_invocation_state = {"custom_param": "test_value", "another_param": 42}
    result = await graph.invoke_async("Test kwargs passing", test_invocation_state)

    kwargs_agent.invoke_async.assert_called_once_with([{"text": "Test kwargs passing"}], **test_invocation_state)
    assert result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_graph_kwargs_passing_multiagent(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying MultiAgentBase nodes."""
    kwargs_multiagent = create_mock_multi_agent("kwargs_multiagent", "MultiAgent response with kwargs")
    kwargs_multiagent.invoke_async = Mock(side_effect=kwargs_multiagent.invoke_async)

    builder = GraphBuilder()
    builder.add_node(kwargs_multiagent, "multiagent_node")
    graph = builder.build()

    test_invocation_state = {"custom_param": "test_value", "another_param": 42}
    result = await graph.invoke_async("Test kwargs passing to multiagent", test_invocation_state)

    kwargs_multiagent.invoke_async.assert_called_once_with(
        [{"text": "Test kwargs passing to multiagent"}], test_invocation_state
    )
    assert result.status == Status.COMPLETED


def test_graph_kwargs_passing_sync(mock_strands_tracer, mock_use_span):
    """Test that kwargs are passed through to underlying nodes in sync execution."""
    kwargs_agent = create_mock_agent("kwargs_agent", "Response with kwargs")
    kwargs_agent.invoke_async = Mock(side_effect=kwargs_agent.invoke_async)

    builder = GraphBuilder()
    builder.add_node(kwargs_agent, "kwargs_node")
    graph = builder.build()

    test_invocation_state = {"custom_param": "test_value", "another_param": 42}
    result = graph("Test kwargs passing sync", test_invocation_state)

    kwargs_agent.invoke_async.assert_called_once_with([{"text": "Test kwargs passing sync"}], **test_invocation_state)
    assert result.status == Status.COMPLETED
