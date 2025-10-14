import pytest

from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.multiagent.base import Status
from strands.multiagent.swarm import Swarm
from strands.types.content import ContentBlock
from tests.fixtures.mock_hook_provider import MockHookProvider


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock implementation
    return f"Results for '{query}': 25% yearly growth assumption, reaching $1.81 trillion by 2030"


@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        return f"The result of {expression} is {eval(expression)}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@pytest.fixture
def hook_provider():
    return MockHookProvider("all")


@pytest.fixture
def researcher_agent(hook_provider):
    """Create an agent specialized in research."""
    return Agent(
        name="researcher",
        system_prompt=(
            "You are a research specialist who excels at finding information. When you need to perform calculations or"
            " format documents, hand off to the appropriate specialist."
        ),
        hooks=[hook_provider],
        tools=[web_search],
    )


@pytest.fixture
def analyst_agent(hook_provider):
    """Create an agent specialized in data analysis."""
    return Agent(
        name="analyst",
        system_prompt=(
            "You are a data analyst who excels at calculations and numerical analysis. When you need"
            " research or document formatting, hand off to the appropriate specialist."
        ),
        hooks=[hook_provider],
        tools=[calculate],
    )


@pytest.fixture
def writer_agent(hook_provider):
    """Create an agent specialized in writing and formatting."""
    return Agent(
        name="writer",
        hooks=[hook_provider],
        system_prompt=(
            "You are a professional writer who excels at formatting and presenting information. When you need research"
            " or calculations, hand off to the appropriate specialist."
        ),
    )


def test_swarm_execution_with_string(researcher_agent, analyst_agent, writer_agent, hook_provider):
    """Test swarm execution with string input."""
    # Create the swarm
    swarm = Swarm([researcher_agent, analyst_agent, writer_agent])

    # Define a task that requires collaboration
    task = (
        "Research the current AI agent market trends, calculate the growth rate assuming 25% yearly growth, "
        "and create a basic report"
    )

    # Execute the swarm
    result = swarm(task)

    # Verify results
    assert result.status.value == "completed"
    assert len(result.results) > 0
    assert result.execution_time > 0
    assert result.execution_count > 0

    # Verify agent history - at least one agent should have been used
    assert len(result.node_history) > 0

    # Just ensure that hooks are emitted; actual content is not verified
    researcher_hooks = hook_provider.extract_for(researcher_agent).event_types_received
    assert BeforeInvocationEvent in researcher_hooks
    assert MessageAddedEvent in researcher_hooks
    assert BeforeModelCallEvent in researcher_hooks
    assert BeforeToolCallEvent in researcher_hooks
    assert AfterToolCallEvent in researcher_hooks
    assert AfterModelCallEvent in researcher_hooks
    assert AfterInvocationEvent in researcher_hooks


@pytest.mark.asyncio
async def test_swarm_execution_with_image(researcher_agent, analyst_agent, writer_agent, yellow_img):
    """Test swarm execution with image input."""
    # Create the swarm
    swarm = Swarm([researcher_agent, analyst_agent, writer_agent])

    # Create content blocks with text and image
    content_blocks: list[ContentBlock] = [
        {"text": "Analyze this image and create a report about what you see:"},
        {"image": {"format": "png", "source": {"bytes": yellow_img}}},
    ]

    # Execute the swarm with multi-modal input
    result = await swarm.invoke_async(content_blocks)

    # Verify results
    assert result.status.value == "completed"
    assert len(result.results) > 0
    assert result.execution_time > 0
    assert result.execution_count > 0

    # Verify agent history - at least one agent should have been used
    assert len(result.node_history) > 0


@pytest.mark.asyncio
async def test_swarm_streaming():
    """Test that Swarm properly streams events during execution."""
    researcher = Agent(
        name="researcher",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a researcher. When you need calculations, hand off to the analyst.",
    )
    analyst = Agent(
        name="analyst",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are an analyst. Use tools to perform calculations.",
        tools=[calculate],
    )

    swarm = Swarm([researcher, analyst])

    # Collect events
    events = []
    async for event in swarm.stream_async("Calculate 10 + 5 and explain the result"):
        events.append(event)

    # Count event categories
    node_start_events = [e for e in events if e.get("multi_agent_node_start")]
    node_stream_events = [e for e in events if e.get("multi_agent_node_stream")]
    result_events = [e for e in events if "result" in e and not e.get("multi_agent_node_stream")]

    # Verify we got multiple events of each type
    assert len(node_start_events) >= 1, f"Expected at least 1 node_start event, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify we have events from at least one agent
    researcher_events = [e for e in events if e.get("node_id") == "researcher"]
    analyst_events = [e for e in events if e.get("node_id") == "analyst"]
    assert len(researcher_events) > 0 or len(analyst_events) > 0, "Expected events from at least one agent"


@pytest.mark.asyncio
async def test_swarm_node_timeout_with_real_streaming():
    """Test that swarm node timeout properly cancels a streaming generator that freezes."""
    import asyncio

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
async def test_swarm_streams_events_before_timeout():
    """Test that swarm events are streamed in real-time before timeout occurs."""
    # Create a normal agent
    agent = Agent(
        name="test_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a test agent. Respond briefly.",
    )

    # Create swarm with reasonable timeout
    swarm = Swarm(
        nodes=[agent],
        max_handoffs=1,
        max_iterations=1,
        node_timeout=30.0,  # Long enough to complete
    )

    # Collect events
    events = []
    async for event in swarm.stream_async("Say hello"):
        events.append(event)

    # Verify we got multiple streaming events before completion
    node_stream_events = [e for e in events if e.get("multi_agent_node_stream")]
    assert len(node_stream_events) > 0, "Expected streaming events before completion"

    # Verify final result - there are 2 result events:
    # 1. Agent's result forwarded as multi_agent_node_stream (with key "result")
    # 2. Swarm's final result (with key "result", not wrapped in node_stream)
    result_events = [e for e in events if "result" in e and not e.get("multi_agent_node_stream")]
    assert len(result_events) >= 1, "Expected at least one result event"

    # The last event should be the swarm result
    final_result = events[-1]["result"]
    assert final_result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_swarm_timeout_cleanup_on_exception():
    """Test that swarm timeout properly cleans up tasks even when exceptions occur."""
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
    # The overall swarm status is COMPLETED even if a node fails
    result = await swarm.invoke_async("Test exception handling")
    # Verify the node failed but swarm completed
    assert "test_agent" in result.results
    assert result.results["test_agent"].status == Status.FAILED


@pytest.mark.asyncio
async def test_swarm_no_timeout_backward_compatibility():
    """Test that swarms without timeout work exactly as before."""
    # Create a normal agent
    agent = Agent(
        name="test_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a test agent. Respond briefly.",
    )

    # Create swarm without timeout (backward compatibility)
    swarm = Swarm(
        nodes=[agent],
        max_handoffs=1,
        max_iterations=1,
    )

    # Note: Swarm has default timeouts for safety
    # This is intentional to prevent runaway executions
    assert swarm.node_timeout == 300.0  # Default node timeout
    assert swarm.execution_timeout == 900.0  # Default execution timeout

    # Execute - should complete normally
    result = await swarm.invoke_async("Say hello")
    assert result.status == Status.COMPLETED


@pytest.mark.asyncio
async def test_swarm_emits_handoff_events():
    """Verify Swarm emits MultiAgentHandoffEvent during streaming."""
    researcher = Agent(
        name="researcher",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are a researcher. When you need calculations, hand off to the analyst.",
    )
    analyst = Agent(
        name="analyst",
        model="us.amazon.nova-pro-v1:0",
        system_prompt="You are an analyst. Use tools to perform calculations.",
        tools=[calculate],
    )

    swarm = Swarm([researcher, analyst])

    # Collect events
    events = []
    async for event in swarm.stream_async("Calculate 10 + 5 and explain the result"):
        events.append(event)

    # Find handoff events
    handoff_events = [e for e in events if e.get("multi_agent_handoff")]

    # Verify we got at least one handoff event
    assert len(handoff_events) > 0, "Expected at least one handoff event"

    # Verify event structure
    handoff = handoff_events[0]
    assert "from_node" in handoff, "Handoff event missing from_node"
    assert "to_node" in handoff, "Handoff event missing to_node"
    assert "message" in handoff, "Handoff event missing message"

    # Verify handoff is from researcher to analyst
    assert handoff["from_node"] == "researcher", f"Expected from_node='researcher', got {handoff['from_node']}"
    assert handoff["to_node"] == "analyst", f"Expected to_node='analyst', got {handoff['to_node']}"


@pytest.mark.asyncio
async def test_swarm_emits_node_complete_events():
    """Verify Swarm emits MultiAgentNodeCompleteEvent after each node."""
    agent = Agent(
        name="test_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a test agent. Respond briefly.",
    )

    swarm = Swarm([agent], max_handoffs=1, max_iterations=1)

    # Collect events
    events = []
    async for event in swarm.stream_async("Say hello"):
        events.append(event)

    # Find node complete events
    complete_events = [e for e in events if e.get("multi_agent_node_complete")]

    # Verify we got at least one node complete event
    assert len(complete_events) > 0, "Expected at least one node complete event"

    # Verify event structure
    complete = complete_events[0]
    assert "node_id" in complete, "Node complete event missing node_id"
    assert "execution_time" in complete, "Node complete event missing execution_time"

    # Verify node_id matches
    assert complete["node_id"] == "test_agent", f"Expected node_id='test_agent', got {complete['node_id']}"

    # Verify execution_time is reasonable
    assert complete["execution_time"] > 0, "Expected positive execution_time"
