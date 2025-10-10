from typing import Any, AsyncIterator

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
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.multiagent.graph import GraphBuilder
from strands.types.content import ContentBlock
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


class CustomStreamingNode(MultiAgentBase):
    """Custom node that wraps an agent and adds custom streaming events."""

    def __init__(self, agent: Agent, name: str):
        self.agent = agent
        self.name = name

    async def invoke_async(
        self, task: str | list[ContentBlock], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        result = await self.agent.invoke_async(task, **kwargs)
        node_result = NodeResult(result=result, status=Status.COMPLETED)
        return MultiAgentResult(status=Status.COMPLETED, results={self.name: node_result})

    async def stream_async(
        self, task: str | list[ContentBlock], invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"custom_event": "start", "node": self.name}
        result = await self.agent.invoke_async(task, **kwargs)
        yield {"custom_event": "agent_complete", "node": self.name}
        node_result = NodeResult(result=result, status=Status.COMPLETED)
        yield {"result": MultiAgentResult(status=Status.COMPLETED, results={self.name: node_result})}


@pytest.mark.asyncio
async def test_graph_streaming_with_agents():
    """Test that Graph properly streams events from agent nodes."""
    math_agent = Agent(
        name="math",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a math assistant.",
        tools=[calculate_sum],
    )
    summary_agent = Agent(
        name="summary",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a summary assistant.",
    )

    builder = GraphBuilder()
    builder.add_node(math_agent, "math")
    builder.add_node(summary_agent, "summary")
    builder.add_edge("math", "summary")
    builder.set_entry_point("math")
    graph = builder.build()

    # Collect events
    events = []
    async for event in graph.stream_async("Calculate 5 + 3 and summarize the result"):
        events.append(event)

    # Count event categories
    node_start_events = [e for e in events if e.get("multi_agent_node_start")]
    node_stream_events = [e for e in events if e.get("multi_agent_node_stream")]
    node_complete_events = [e for e in events if e.get("multi_agent_node_complete")]
    result_events = [e for e in events if "result" in e and "multi_agent_node_start" not in e]

    # Verify we got multiple events of each type
    assert len(node_start_events) >= 2, f"Expected at least 2 node_start events, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(node_complete_events) >= 2, f"Expected at least 2 node_complete events, got {len(node_complete_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify we have events for both nodes
    math_events = [e for e in events if e.get("node_id") == "math"]
    summary_events = [e for e in events if e.get("node_id") == "summary"]
    assert len(math_events) > 0, "Expected events from math node"
    assert len(summary_events) > 0, "Expected events from summary node"


@pytest.mark.asyncio
async def test_graph_streaming_with_custom_node():
    """Test that Graph properly streams events from custom MultiAgentBase nodes."""
    math_agent = Agent(
        name="math",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a math assistant.",
        tools=[calculate_sum],
    )
    summary_agent = Agent(
        name="summary",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a summary assistant.",
    )

    # Create a custom node
    custom_node = CustomStreamingNode(summary_agent, "custom_summary")

    builder = GraphBuilder()
    builder.add_node(math_agent, "math")
    builder.add_node(custom_node, "custom_summary")
    builder.add_edge("math", "custom_summary")
    builder.set_entry_point("math")
    graph = builder.build()

    # Collect events
    events = []
    async for event in graph.stream_async("Calculate 5 + 3 and summarize the result"):
        events.append(event)

    # Count event categories
    node_start_events = [e for e in events if e.get("multi_agent_node_start")]
    node_stream_events = [e for e in events if e.get("multi_agent_node_stream")]
    custom_events = [e for e in events if e.get("custom_event")]
    result_events = [e for e in events if "result" in e and "multi_agent_node_start" not in e]

    # Verify we got multiple events of each type
    assert len(node_start_events) >= 2, f"Expected at least 2 node_start events, got {len(node_start_events)}"
    assert len(node_stream_events) > 5, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(custom_events) >= 2, f"Expected at least 2 custom events (start, complete), got {len(custom_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify custom events are properly structured
    custom_start = [e for e in custom_events if e.get("custom_event") == "start"]
    custom_complete = [e for e in custom_events if e.get("custom_event") == "agent_complete"]

    assert len(custom_start) >= 1, "Expected at least 1 custom start event"
    assert len(custom_complete) >= 1, "Expected at least 1 custom complete event"


@pytest.mark.asyncio
async def test_nested_graph_streaming():
    """Test that nested graphs properly propagate streaming events."""
    math_agent = Agent(
        name="math",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a math assistant.",
        tools=[calculate_sum],
    )
    analysis_agent = Agent(
        name="analysis",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are an analysis assistant.",
    )

    # Create nested graph
    nested_builder = GraphBuilder()
    nested_builder.add_node(math_agent, "calculator")
    nested_builder.add_node(analysis_agent, "analyzer")
    nested_builder.add_edge("calculator", "analyzer")
    nested_builder.set_entry_point("calculator")
    nested_graph = nested_builder.build()

    # Create outer graph with nested graph
    summary_agent = Agent(
        name="summary",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a summary assistant.",
    )

    outer_builder = GraphBuilder()
    outer_builder.add_node(nested_graph, "computation")
    outer_builder.add_node(summary_agent, "summary")
    outer_builder.add_edge("computation", "summary")
    outer_builder.set_entry_point("computation")
    outer_graph = outer_builder.build()

    # Collect events
    events = []
    async for event in outer_graph.stream_async("Calculate 7 + 8 and provide a summary"):
        events.append(event)

    # Count event categories
    node_start_events = [e for e in events if e.get("multi_agent_node_start")]
    node_stream_events = [e for e in events if e.get("multi_agent_node_stream")]
    result_events = [e for e in events if "result" in e and "multi_agent_node_start" not in e]

    # Verify we got multiple events
    assert len(node_start_events) >= 2, f"Expected at least 2 node_start events, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify we have events from nested nodes
    computation_events = [e for e in events if e.get("node_id") == "computation"]
    summary_events = [e for e in events if e.get("node_id") == "summary"]
    assert len(computation_events) > 0, "Expected events from computation (nested graph) node"
    assert len(summary_events) > 0, "Expected events from summary node"
