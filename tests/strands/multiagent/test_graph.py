from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands.agent import Agent, AgentResult
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult
from strands.multiagent.graph import GraphBuilder, GraphEdge, GraphNode, GraphResult, GraphState, Status


def create_mock_agent(name, response_text="Default response", metrics=None, agent_id=None):
    """Create a mock Agent with specified properties."""
    agent = Mock(spec=Agent)
    agent.name = name
    agent.id = agent_id or f"{name}_id"

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
    return agent


def create_mock_multi_agent(name, response_text="Multi-agent response"):
    """Create a mock MultiAgentBase with specified properties."""
    multi_agent = Mock(spec=MultiAgentBase)
    multi_agent.name = name
    multi_agent.id = f"{name}_id"

    mock_node_result = NodeResult(
        results=AgentResult(
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
        execution_time=150.0,
    )
    multi_agent.execute = AsyncMock(return_value=mock_result)
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
def mock_graph(mock_agents, string_content_agent):
    """Create a graph for testing various scenarios."""

    def condition_check_completion(state: GraphState) -> bool:
        return "start_agent" in state.completed_nodes

    def always_false_condition(state: GraphState) -> bool:
        return False

    builder = GraphBuilder()

    # Add nodes
    builder.add_node(mock_agents["start_agent"], "start_agent")
    builder.add_node(mock_agents["multi_agent"], "multi_node")
    builder.add_node(mock_agents["conditional_agent"], "conditional_agent")
    builder.add_node(mock_agents["final_agent"], "final_node")
    builder.add_node(mock_agents["no_metrics_agent"], "no_metrics_node")
    builder.add_node(mock_agents["partial_metrics_agent"], "partial_metrics_node")
    builder.add_node(string_content_agent, "string_content_node")
    builder.add_node(mock_agents["blocked_agent"], "blocked_node")

    # Add edges
    builder.add_edge("start_agent", "multi_node")
    builder.add_edge("start_agent", "conditional_agent", condition=condition_check_completion)
    builder.add_edge("multi_node", "final_node")
    builder.add_edge("conditional_agent", "final_node")
    builder.add_edge("start_agent", "no_metrics_node")
    builder.add_edge("start_agent", "partial_metrics_node")
    builder.add_edge("start_agent", "string_content_node")
    builder.add_edge("start_agent", "blocked_node", condition=always_false_condition)

    builder.set_entry_point("start_agent")
    return builder.build()


@pytest.mark.asyncio
async def test_graph_execution(mock_graph, mock_agents, string_content_agent):
    """Test comprehensive graph execution with diverse nodes and conditional edges."""
    # Test graph structure
    assert len(mock_graph.nodes) == 8
    assert len(mock_graph.edges) == 8
    assert len(mock_graph.entry_points) == 1
    assert "start_agent" in mock_graph.entry_points

    # Test node properties
    start_node = mock_graph.nodes["start_agent"]
    assert start_node.node_id == "start_agent"
    assert start_node.executor == mock_agents["start_agent"]
    assert start_node.status == Status.PENDING
    assert len(start_node.dependencies) == 0

    # Test conditional edge evaluation
    conditional_edge = next(
        edge for edge in mock_graph.edges if edge.from_node == "start_agent" and edge.to_node == "conditional_agent"
    )
    assert conditional_edge.condition is not None
    assert not conditional_edge.should_traverse(GraphState())
    assert conditional_edge.should_traverse(GraphState(completed_nodes={"start_agent"}))

    # Test graph execution with mocked timer
    with patch("strands.multiagent.graph.time.time") as mock_time:
        times = [0.0, 5.0, 15.0, 18.0, 28.0, 30.0, 35.0, 36.0, 40.0, 42.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
        mock_time.side_effect = times

        result = await mock_graph.execute("Test comprehensive execution")

        # Verify execution results
        assert result.status == Status.COMPLETED
        assert result.total_nodes == 8
        assert result.completed_nodes == 7  # All except blocked_node
        assert result.failed_nodes == 0
        assert len(result.execution_order) == 7
        assert result.execution_order[0] == "start_agent"

        # Verify agent calls
        mock_agents["start_agent"].assert_called_once()
        mock_agents["multi_agent"].execute.assert_called_once()
        mock_agents["conditional_agent"].assert_called_once()
        mock_agents["final_agent"].assert_called_once()
        mock_agents["no_metrics_agent"].assert_called_once()
        mock_agents["partial_metrics_agent"].assert_called_once()
        string_content_agent.assert_called_once()
        mock_agents["blocked_agent"].assert_not_called()

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
        assert result.entry_points == ["start_agent"]


@pytest.mark.asyncio
async def test_graph_unsupported_node_type():
    """Test unsupported executor type error handling."""

    class UnsupportedExecutor:
        pass

    builder = GraphBuilder()
    builder.add_node(UnsupportedExecutor(), "unsupported_node")
    graph = builder.build()

    with pytest.raises(ValueError, match="Node 'unsupported_node' of type.*is not supported"):
        await graph.execute("test task")


@pytest.mark.asyncio
async def test_graph_execution_with_failures():
    """Test graph execution error handling and failure propagation."""
    failing_agent = Mock(spec=Agent)
    failing_agent.name = "failing_agent"
    failing_agent.id = "fail_node"
    failing_agent.side_effect = Exception("Simulated failure")
    failing_agent.__call__ = Mock(side_effect=Exception("Simulated failure"))

    success_agent = create_mock_agent("success_agent", "Success")

    builder = GraphBuilder()
    builder.add_node(failing_agent, "fail_node")
    builder.add_node(success_agent, "success_node")
    builder.add_edge("fail_node", "success_node")
    builder.set_entry_point("fail_node")

    graph = builder.build()

    with pytest.raises(Exception, match="Simulated failure"):
        await graph.execute("Test error handling")

    assert graph.state.status == Status.FAILED
    assert "fail_node" in graph.state.failed_nodes
    assert len(graph.state.completed_nodes) == 0


@pytest.mark.asyncio
async def test_graph_edge_cases():
    """Test specific edge cases for coverage."""
    # Test entry node execution without dependencies
    entry_agent = create_mock_agent("entry_agent", "Entry response")

    builder = GraphBuilder()
    builder.add_node(entry_agent, "entry_only")
    graph = builder.build()

    result = await graph.execute("Original task")

    # Verify entry node was called with original task
    entry_agent.assert_called_once_with("Original task")
    assert result.status == Status.COMPLETED


def test_graph_string_representation():
    """Test string representation with nested structures."""
    mock_multi_with_agents = Mock(spec=MultiAgentBase)
    mock_multi_with_agents.agents = [Mock(name="agent1"), Mock(name="agent2")]

    builder = GraphBuilder()
    builder.add_node(mock_multi_with_agents, "multi_with_agents")
    graph = builder.build()

    graph_str = str(graph)
    assert "multi_with_agents (Mock)" in graph_str
    assert "└─" in graph_str  # Tests nested agent display


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
    builder.entry_points.add("invalid1")
    builder.entry_points.add("invalid2")
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
    assert graph.entry_points == {"entry"}

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
    edge_simple = GraphEdge("a", "b")
    assert edge_simple.from_node == "a"
    assert edge_simple.to_node == "b"
    assert edge_simple.condition is None
    assert edge_simple.should_traverse(GraphState())

    def test_condition(state):
        return len(state.completed_nodes) > 0

    edge_conditional = GraphEdge("a", "b", condition=test_condition)
    assert edge_conditional.condition is not None
    assert not edge_conditional.should_traverse(GraphState())
    assert edge_conditional.should_traverse(GraphState(completed_nodes={"some_node"}))

    # Test GraphEdge hashing
    edge1 = GraphEdge("x", "y")
    edge2 = GraphEdge("x", "y")
    edge3 = GraphEdge("y", "x")
    assert hash(edge1) == hash(edge2)
    assert hash(edge1) != hash(edge3)

    # Test GraphNode initialization and ready check
    mock_agent = create_mock_agent("test_agent")
    node = GraphNode("test_node", mock_agent)
    assert node.node_id == "test_node"
    assert node.executor == mock_agent
    assert node.status == Status.PENDING
    assert len(node.dependencies) == 0
    assert node.is_ready(set())

    node_with_deps = GraphNode("dependent_node", mock_agent, dependencies={"dep1", "dep2"})
    assert not node_with_deps.is_ready({"dep1"})
    assert node_with_deps.is_ready({"dep1", "dep2"})
    assert node_with_deps.is_ready({"dep1", "dep2", "extra"})


@pytest.mark.asyncio
async def test_graph_resume_not_implemented():
    """Test that graph resume method raises NotImplementedError."""
    builder = GraphBuilder()
    mock_agent = create_mock_agent("test_agent")
    builder.add_node(mock_agent, "test_node")
    graph = builder.build()

    with pytest.raises(NotImplementedError, match="resume not implemented"):
        await graph.resume("test task", GraphState())
