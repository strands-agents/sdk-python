from typing import Any, AsyncIterator, Literal
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

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
from strands.session.file_session_manager import FileSessionManager
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
async def test_graph_streaming_with_agents(alist):
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
    builder.set_node_timeout(900.0)  # Verify timeout doesn't interfere with streaming
    graph = builder.build()

    # Collect events
    events = await alist(graph.stream_async("Calculate 5 + 3 and summarize the result"))

    # Count event categories
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    node_stop_events = [e for e in events if e.get("type") == "multiagent_node_stop"]
    handoff_events = [e for e in events if e.get("type") == "multiagent_handoff"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Verify we got multiple events of each type
    assert len(node_start_events) >= 2, f"Expected at least 2 node_start events, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(node_stop_events) >= 2, f"Expected at least 2 node_stop events, got {len(node_stop_events)}"
    assert len(handoff_events) >= 1, f"Expected at least 1 handoff event, got {len(handoff_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify handoff event structure
    handoff = handoff_events[0]
    assert "from_node_ids" in handoff, "Handoff event missing from_node_ids"
    assert "to_node_ids" in handoff, "Handoff event missing to_node_ids"
    assert isinstance(handoff["from_node_ids"], list), "from_node_ids should be a list"
    assert isinstance(handoff["to_node_ids"], list), "to_node_ids should be a list"
    assert "math" in handoff["from_node_ids"], "Expected math in from_node_ids"
    assert "summary" in handoff["to_node_ids"], "Expected summary in to_node_ids"

    # Verify we have events for both nodes
    math_events = [e for e in events if e.get("node_id") == "math"]
    summary_events = [e for e in events if e.get("node_id") == "summary"]
    assert len(math_events) > 0, "Expected events from math node"
    assert len(summary_events) > 0, "Expected events from summary node"


@pytest.mark.asyncio
async def test_graph_streaming_with_custom_node(alist):
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
    events = await alist(graph.stream_async("Calculate 5 + 3 and summarize the result"))

    # Count event categories
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Extract custom events from wrapped node_stream events
    # Structure: {"type": "multiagent_node_stream", "node_id": "...", "event": {...}}
    custom_events = []
    for e in node_stream_events:
        if e.get("type") == "multiagent_node_stream" and "event" in e:
            inner_event = e["event"]
            if isinstance(inner_event, dict) and "custom_event" in inner_event:
                custom_events.append(inner_event)

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
async def test_nested_graph_streaming(alist):
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
    events = await alist(outer_graph.stream_async("Calculate 7 + 8 and provide a summary"))

    # Count event categories
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Verify we got multiple events
    assert len(node_start_events) >= 2, f"Expected at least 2 node_start events, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify we have events from nested nodes
    computation_events = [e for e in events if e.get("node_id") == "computation"]
    summary_events = [e for e in events if e.get("node_id") == "summary"]
    assert len(computation_events) > 0, "Expected events from computation (nested graph) node"
    assert len(summary_events) > 0, "Expected events from summary node"


@pytest.mark.asyncio
async def test_graph_metrics_accumulation():
    """Test that graph properly accumulates metrics from agent nodes."""
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

    result = await graph.invoke_async("Calculate 5 + 3 and summarize the result")

    # Verify result has accumulated metrics
    assert result.accumulated_usage is not None
    assert result.accumulated_usage["totalTokens"] > 0, "Expected non-zero total tokens"
    assert result.accumulated_usage["inputTokens"] > 0, "Expected non-zero input tokens"
    assert result.accumulated_usage["outputTokens"] > 0, "Expected non-zero output tokens"

    assert result.accumulated_metrics is not None
    assert result.accumulated_metrics["latencyMs"] > 0, "Expected non-zero latency"

    # Verify individual node results have metrics
    for node_id, node_result in result.results.items():
        assert node_result.accumulated_usage is not None, f"Node {node_id} missing usage metrics"
        assert node_result.accumulated_usage["totalTokens"] > 0, f"Node {node_id} has zero total tokens"
        assert node_result.accumulated_metrics is not None, f"Node {node_id} missing metrics"

    # Verify accumulated metrics are sum of node metrics
    total_tokens = sum(node_result.accumulated_usage["totalTokens"] for node_result in result.results.values())
    assert result.accumulated_usage["totalTokens"] == total_tokens, "Accumulated tokens don't match sum of node tokens"


@pytest.mark.asyncio
async def test_graph_interrupt_and_resume():
    """Test graph interruption and resume functionality with FileSessionManager."""

    session_id = str(uuid4())

    # Create real agents
    agent1 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 1", name="agent1")
    agent2 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 2", name="agent2")
    agent3 = Agent(model="us.amazon.nova-pro-v1:0", system_prompt="You are agent 3", name="agent3")

    session_manager = FileSessionManager(session_id=session_id)

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
    async def failing_stream_async(*args, **kwargs):
        raise Exception("Simulated failure in agent2")
        yield  # This line is never reached, but makes it an async generator

    with patch.object(agent2, "stream_async", side_effect=failing_stream_async):
        try:
            await graph.invoke_async("This is a test task, just do it shortly")
            raise AssertionError("Expected exception was not raised")
        except Exception as e:
            assert "Simulated failure in agent2" in str(e)

    # Verify partial execution was persisted
    persisted_state = session_manager.read_multi_agent(session_id, graph.id)
    assert persisted_state is not None
    assert persisted_state["type"] == "graph"
    assert persisted_state["status"] == "failed"
    assert len(persisted_state["completed_nodes"]) == 1  # Only node1 completed
    assert "node1" in persisted_state["completed_nodes"]
    assert "node2" in persisted_state["next_nodes_to_execute"]
    assert "node2" in persisted_state["failed_nodes"]

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

    final_state = session_manager.read_multi_agent(session_id, graph.id)
    assert final_state["status"] == "completed"
    assert len(final_state["completed_nodes"]) == 3

    # Clean up
    session_manager.delete_session(session_id)


@pytest.mark.asyncio
async def test_graph_structured_output_conditional_routing():
    """Test conditional edge traversal based on structured output from a node."""

    # Classifier's structured output for routing decisions
    class ClassificationResult(BaseModel):
        category: Literal["technical", "billing", "general"]
        confidence: float
        reasoning: str

    # Specialist agents have their own structured output models
    class TechnicalResponse(BaseModel):
        issue_type: str
        resolution: str

    class BillingResponse(BaseModel):
        issue_type: str
        refund_eligible: bool

    class GeneralResponse(BaseModel):
        answer: str

    # Classifier agent uses ClassificationResult for routing
    classifier_agent = Agent(
        name="classifier",
        model="us.amazon.nova-pro-v1:0",
        system_prompt=(
            "You are a customer support classifier. Classify queries into:\n"
            "- technical: software issues, bugs, crashes\n"
            "- billing: payments, charges, refunds\n"
            "- general: hours, contact info, policies"
        ),
        structured_output_model=ClassificationResult,
    )

    # Specialist agents - each has their OWN structured output model
    technical_agent = Agent(
        name="technical_support",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are technical support. Diagnose the issue and provide resolution steps.",
        structured_output_model=TechnicalResponse,
    )

    billing_agent = Agent(
        name="billing_support",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are billing support. Analyze the billing issue and determine refund eligibility.",
        structured_output_model=BillingResponse,
    )

    general_agent = Agent(
        name="general_support",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are general support. Answer general inquiries helpfully.",
        structured_output_model=GeneralResponse,
    )

    # Conditional edge functions that read classifier's structured output
    def route_to_technical(state):
        classifier_result = state.results.get("classifier")
        if classifier_result and classifier_result.result:
            structured = classifier_result.result.structured_output
            return structured and structured.category == "technical"
        return False

    def route_to_billing(state):
        classifier_result = state.results.get("classifier")
        if classifier_result and classifier_result.result:
            structured = classifier_result.result.structured_output
            return structured and structured.category == "billing"
        return False

    def route_to_general(state):
        classifier_result = state.results.get("classifier")
        if classifier_result and classifier_result.result:
            structured = classifier_result.result.structured_output
            return structured and structured.category == "general"
        return False

    # Build the graph with conditional routing
    builder = GraphBuilder()
    builder.add_node(classifier_agent, "classifier")
    builder.add_node(technical_agent, "technical")
    builder.add_node(billing_agent, "billing")
    builder.add_node(general_agent, "general")

    # Conditional edges from classifier to specialists
    builder.add_edge("classifier", "technical", condition=route_to_technical)
    builder.add_edge("classifier", "billing", condition=route_to_billing)
    builder.add_edge("classifier", "general", condition=route_to_general)

    builder.set_entry_point("classifier")
    graph = builder.build()

    # Test: Technical query should route to technical_support
    technical_query = "My application keeps crashing when I upload large files"
    result = await graph.invoke_async(technical_query)

    assert result.status == Status.COMPLETED
    assert "classifier" in result.results

    # Verify classifier's structured output was captured
    classifier_result = result.results["classifier"]
    assert classifier_result.result.structured_output is not None
    assert isinstance(classifier_result.result.structured_output, ClassificationResult)
    assert classifier_result.result.structured_output.category == "technical"

    # Verify only technical agent was executed (not billing or general)
    assert "technical" in result.results
    assert "billing" not in result.results
    assert "general" not in result.results

    # Verify technical agent's own structured output was preserved
    technical_result = result.results["technical"]
    assert technical_result.result.structured_output is not None
    assert isinstance(technical_result.result.structured_output, TechnicalResponse)


@pytest.mark.asyncio
async def test_self_loop_resume_from_persisted_state(tmp_path):
    """Test resuming self-loop from persisted state where next node is itself."""

    session_id = f"self_loop_resume_{uuid4()}"
    session_manager = FileSessionManager(session_id=session_id, storage_dir=str(tmp_path))

    counter_agent = Agent(
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a counter. Just respond with 'Count: 1', 'Count: 2', Stop at 5.",
    )

    def should_continue_loop(state):
        loop_executions = len([node for node in state.execution_order if node.node_id == "loop_node"])
        return loop_executions < 5

    builder = GraphBuilder()
    builder.add_node(counter_agent, "loop_node")
    builder.add_edge("loop_node", "loop_node", condition=should_continue_loop)
    builder.set_entry_point("loop_node")
    builder.set_session_manager(session_manager)
    builder.reset_on_revisit(True)

    graph = builder.build()

    call_count = 0
    original_stream = counter_agent.stream_async

    async def failing_after_two(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            async for event in original_stream(*args, **kwargs):
                yield event
        else:
            raise Exception("Simulated failure after two executions")

    with patch.object(counter_agent, "stream_async", side_effect=failing_after_two):
        try:
            await graph.invoke_async("Count till 5")
        except Exception as e:
            assert "Simulated failure after two executions" in str(e)

    persisted_state = session_manager.read_multi_agent(session_id, graph.id)
    assert persisted_state["status"] == "failed"
    assert "loop_node" in persisted_state.get("failed_nodes")
    assert len(persisted_state.get("execution_order")) == 2

    result = await graph.invoke_async("Continue counting to 5")
    assert result.status == Status.COMPLETED
    assert len(result.execution_order) == 5
    assert all(node.node_id == "loop_node" for node in result.execution_order)
