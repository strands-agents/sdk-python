from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from strands.agent import Agent, AgentResult
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

    with pytest.raises(ValueError, match="Node 'unsupported_node' of type.*is not supported"):
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

    with pytest.raises(Exception, match="Simulated failure"):
        await graph.invoke_async("Test error handling")

    assert graph.state.status == Status.FAILED
    assert any(node.node_id == "fail_node" for node in graph.state.failed_nodes)
    assert len(graph.state.completed_nodes) == 0
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
    from strands.multiagent.graph import Graph, GraphNode

    duplicate_agent = create_mock_agent("duplicate_agent")
    node1 = GraphNode("node1", duplicate_agent)
    node2 = GraphNode("node2", duplicate_agent)  # Same agent instance
    nodes = {"node1": node1, "node2": node2}
    with pytest.raises(ValueError, match="Duplicate node instance detected"):
        Graph(nodes=nodes, edges=set(), entry_points=set())

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

    # Test cycle detection
    builder = GraphBuilder()
    builder.add_node(agent1, "a")
    builder.add_node(agent2, "b")
    builder.add_node(create_mock_agent("agent3"), "c")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "a")  # Creates cycle
    builder.set_entry_point("a")

    with pytest.raises(ValueError, match="Graph contains cycles"):
        builder.build()

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

    agent_with_hooks = create_mock_agent("agent_with_hooks")
    agent_with_hooks._session_manager = None
    agent_with_hooks.hooks = HookRegistry()
    agent_with_hooks.hooks.add_hook(TestHookProvider())

    builder = GraphBuilder()
    with pytest.raises(ValueError, match="Agent callbacks are not supported for Graph agents yet"):
        builder.add_node(agent_with_hooks)

    # Test validation in Graph constructor (when nodes are passed directly)
    # Test with session manager in Graph constructor
    node_with_session = GraphNode("node_with_session", agent_with_session)
    with pytest.raises(ValueError, match="Session persistence is not supported for Graph agents yet"):
        Graph(nodes={"node_with_session": node_with_session}, edges=set(), entry_points=set())

    # Test with callbacks in Graph constructor
    node_with_hooks = GraphNode("node_with_hooks", agent_with_hooks)
    with pytest.raises(ValueError, match="Agent callbacks are not supported for Graph agents yet"):
        Graph(nodes={"node_with_hooks": node_with_hooks}, edges=set(), entry_points=set())
