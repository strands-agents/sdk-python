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
async def test_swarm_streaming(alist):
    """Test that Swarm properly streams all event types during execution."""
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

    swarm = Swarm([researcher, analyst], node_timeout=900.0)

    # Collect events
    events = await alist(swarm.stream_async("Calculate 10 + 5 and explain the result"))

    # Count event categories
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    node_stop_events = [e for e in events if e.get("type") == "multiagent_node_stop"]
    handoff_events = [e for e in events if e.get("type") == "multiagent_handoff"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Verify we got multiple events of each type
    assert len(node_start_events) >= 1, f"Expected at least 1 node_start event, got {len(node_start_events)}"
    assert len(node_stream_events) > 10, f"Expected many node_stream events, got {len(node_stream_events)}"
    assert len(node_stop_events) >= 1, f"Expected at least 1 node_stop event, got {len(node_stop_events)}"
    assert len(handoff_events) >= 1, f"Expected at least 1 handoff event, got {len(handoff_events)}"
    assert len(result_events) >= 1, f"Expected at least 1 result event, got {len(result_events)}"

    # Verify handoff event structure
    handoff = handoff_events[0]
    assert "from_nodes" in handoff, "Handoff event missing from_nodes"
    assert "to_nodes" in handoff, "Handoff event missing to_nodes"
    assert "message" in handoff, "Handoff event missing message"
    assert handoff["from_nodes"] == ["researcher"], f"Expected from_nodes=['researcher'], got {handoff['from_nodes']}"
    assert handoff["to_nodes"] == ["analyst"], f"Expected to_nodes=['analyst'], got {handoff['to_nodes']}"

    # Verify node stop event structure
    stop_event = node_stop_events[0]
    assert "node_id" in stop_event, "Node stop event missing node_id"
    assert "node_result" in stop_event, "Node stop event missing node_result"
    node_result = stop_event["node_result"]
    assert hasattr(node_result, "execution_time"), "NodeResult missing execution_time"
    assert node_result.execution_time > 0, "Expected positive execution_time"

    # Verify we have events from at least one agent
    researcher_events = [e for e in events if e.get("node_id") == "researcher"]
    analyst_events = [e for e in events if e.get("node_id") == "analyst"]
    assert len(researcher_events) > 0 or len(analyst_events) > 0, "Expected events from at least one agent"
