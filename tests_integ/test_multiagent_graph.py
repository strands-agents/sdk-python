import tempfile
from unittest.mock import patch
from uuid import uuid4

import pytest

from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    MessageAddedEvent,
)
from strands.multiagent.base import Status
from strands.multiagent.graph import GraphBuilder
from strands.session.file_session_manager import FileSessionManager
from strands.types.content import ContentBlock
from strands.types.session import SessionType
from tests.fixtures.mock_hook_provider import MockHookProvider


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


@tool
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers together."""
    return x * y


@pytest.fixture
def hook_provider():
    return MockHookProvider("all")


@pytest.fixture
def math_agent(hook_provider):
    """Create an agent specialized in mathematical operations."""
    return Agent(
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a mathematical assistant. Always provide clear, step-by-step calculations.",
        hooks=[hook_provider],
        tools=[calculate_sum, multiply_numbers],
    )


@pytest.fixture
def analysis_agent(hook_provider):
    """Create an agent specialized in data analysis."""
    return Agent(
        model="us.amazon.nova-pro-v1:0",
        hooks=[hook_provider],
        system_prompt="You are a data analysis expert. Provide insights and interpretations of numerical results.",
    )


@pytest.fixture
def summary_agent(hook_provider):
    """Create an agent specialized in summarization."""
    return Agent(
        model="us.amazon.nova-lite-v1:0",
        hooks=[hook_provider],
        system_prompt="You are a summarization expert. Create concise, clear summaries of complex information.",
    )


@pytest.fixture
def validation_agent(hook_provider):
    """Create an agent specialized in validation."""
    return Agent(
        model="us.amazon.nova-pro-v1:0",
        hooks=[hook_provider],
        system_prompt="You are a validation expert. Check results for accuracy and completeness.",
    )


@pytest.fixture
def image_analysis_agent(hook_provider):
    """Create an agent specialized in image analysis."""
    return Agent(
        hooks=[hook_provider],
        system_prompt=(
            "You are an image analysis expert. Describe what you see in images and provide detailed analysis."
        ),
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def nested_computation_graph(math_agent, analysis_agent):
    """Create a nested graph for mathematical computation and analysis."""
    builder = GraphBuilder()

    # Add agents to nested graph
    builder.add_node(math_agent, "calculator")
    builder.add_node(analysis_agent, "analyzer")

    # Connect them sequentially
    builder.add_edge("calculator", "analyzer")
    builder.set_entry_point("calculator")

    return builder.build()


@pytest.mark.asyncio
async def test_graph_execution_with_string(math_agent, summary_agent, validation_agent, nested_computation_graph):
    # Define conditional functions
    def should_validate(state):
        """Condition to determine if validation should run."""
        return any(node.node_id == "computation_subgraph" for node in state.completed_nodes)

    def proceed_to_second_summary(state):
        """Condition to skip additional summary."""
        return False  # Skip for this test

    builder = GraphBuilder()

    summary_agent_duplicate = Agent(
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a summarization expert. Create concise, clear summaries of complex information.",
    )

    # Add various node types
    builder.add_node(nested_computation_graph, "computation_subgraph")  # Nested Graph node
    builder.add_node(math_agent, "secondary_math")  # Agent node
    builder.add_node(validation_agent, "validator")  # Agent node with condition
    builder.add_node(summary_agent, "primary_summary")  # Agent node
    builder.add_node(summary_agent_duplicate, "secondary_summary")  # Another Agent node

    # Add edges with various configurations
    builder.add_edge("computation_subgraph", "secondary_math")  # Graph -> Agent
    builder.add_edge("computation_subgraph", "validator", condition=should_validate)  # Conditional edge
    builder.add_edge("secondary_math", "primary_summary")  # Agent -> Agent
    builder.add_edge("validator", "primary_summary")  # Agent -> Agent
    builder.add_edge("primary_summary", "secondary_summary", condition=proceed_to_second_summary)  # Conditional (false)

    builder.set_entry_point("computation_subgraph")

    graph = builder.build()

    task = (
        "Calculate 15 + 27 and 8 * 6, analyze both results, perform additional calculations, validate everything, "
        "and provide a comprehensive summary"
    )
    result = await graph.invoke_async(task)

    # Verify results
    assert result.status.value == "completed"
    assert result.total_nodes == 5
    assert result.completed_nodes == 4  # All except secondary_summary (blocked by false condition)
    assert result.failed_nodes == 0
    assert len(result.results) == 4

    # Verify execution order - extract node_ids from GraphNode objects
    execution_order_ids = [node.node_id for node in result.execution_order]
    # With parallel execution, secondary_math and validator can complete in any order
    assert execution_order_ids[0] == "computation_subgraph"  # First
    assert execution_order_ids[3] == "primary_summary"  # Last
    assert set(execution_order_ids[1:3]) == {"secondary_math", "validator"}  # Middle two in any order

    # Verify specific nodes completed
    assert "computation_subgraph" in result.results
    assert "secondary_math" in result.results
    assert "validator" in result.results
    assert "primary_summary" in result.results
    assert "secondary_summary" not in result.results  # Should be blocked by condition

    # Verify nested graph execution
    nested_result = result.results["computation_subgraph"].result
    assert nested_result.status.value == "completed"


@pytest.mark.asyncio
async def test_graph_execution_with_image(image_analysis_agent, summary_agent, yellow_img, hook_provider):
    """Test graph execution with multi-modal image input."""
    builder = GraphBuilder()

    # Add agents to graph
    builder.add_node(image_analysis_agent, "image_analyzer")
    builder.add_node(summary_agent, "summarizer")

    # Connect them sequentially
    builder.add_edge("image_analyzer", "summarizer")
    builder.set_entry_point("image_analyzer")

    graph = builder.build()

    # Create content blocks with text and image
    content_blocks: list[ContentBlock] = [
        {"text": "Analyze this image and describe what you see:"},
        {"image": {"format": "png", "source": {"bytes": yellow_img}}},
    ]

    # Execute the graph with multi-modal input
    result = await graph.invoke_async(content_blocks)

    # Verify results
    assert result.status.value == "completed"
    assert result.total_nodes == 2
    assert result.completed_nodes == 2
    assert result.failed_nodes == 0
    assert len(result.results) == 2

    # Verify execution order
    execution_order_ids = [node.node_id for node in result.execution_order]
    assert execution_order_ids == ["image_analyzer", "summarizer"]

    # Verify both nodes completed
    assert "image_analyzer" in result.results
    assert "summarizer" in result.results

    expected_hook_events = [
        AgentInitializedEvent,
        BeforeInvocationEvent,
        MessageAddedEvent,
        BeforeModelCallEvent,
        AfterModelCallEvent,
        MessageAddedEvent,
        AfterInvocationEvent,
    ]

    assert hook_provider.extract_for(image_analysis_agent).event_types_received == expected_hook_events
    assert hook_provider.extract_for(summary_agent).event_types_received == expected_hook_events


@pytest.mark.asyncio
async def test_graph_interrupt_and_resume():
    """Test graph interruption and resume functionality with FileSessionManager."""

    with tempfile.TemporaryDirectory() as temp_dir:
        session_id = str(uuid4())

        # Create real agents
        agent1 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 1", name="agent1")
        agent2 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 2", name="agent2")
        agent3 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 3", name="agent3")

        session_manager = FileSessionManager(
            session_id=session_id, storage_dir=temp_dir, session_type=SessionType.MULTI_AGENT
        )

        builder = GraphBuilder()
        builder.add_node(agent1, "node1")
        builder.add_node(agent2, "node2")
        builder.add_node(agent3, "node3")
        builder.add_edge("node1", "node2")
        builder.add_edge("node2", "node3")
        builder.set_entry_point("node1")
        builder.set_session_manager(session_manager)

        graph = builder.build()

        # Mock agent2 to fail on first execution
        async def failing_invoke(*args, **kwargs):
            raise Exception("Simulated failure in agent2")

        with patch.object(agent2, "invoke_async", side_effect=failing_invoke):
            # First execution - should fail at agent2
            try:
                await graph.invoke_async("Test task")
            except Exception as e:
                assert "Simulated failure in agent2" in str(e)

        # Verify partial execution was persisted
        persisted_state = session_manager.read_multi_agent_json()
        assert persisted_state is not None
        assert persisted_state["type"] == "graph"
        assert persisted_state["status"] == "failed"
        assert len(persisted_state["completed_nodes"]) == 1  # Only node1 completed
        assert "node1" in persisted_state["completed_nodes"]
        assert "node2" in persisted_state["next_node_to_execute"]

        # Track execution count before resume
        initial_execution_count = graph.state.execution_count

        # Execute graph again
        result = await graph.invoke_async("Test task")

        # Verify successful completion
        assert result.status == Status.COMPLETED
        assert len(result.results) == 3

        execution_order_ids = [node.node_id for node in result.execution_order]
        assert execution_order_ids == ["node1", "node2", "node3"]

        # Verify only 2 additional nodes were executed
        assert result.execution_count == initial_execution_count + 2

        final_state = session_manager.read_multi_agent_json()
        assert final_state["status"] == "completed"
        assert len(final_state["completed_nodes"]) == 3

        # Clean up
        session_manager.delete_session(session_id)
