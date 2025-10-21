import tempfile
from unittest.mock import patch
from uuid import uuid4

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
from strands.session.file_session_manager import FileSessionManager
from strands.types.content import ContentBlock
from strands.types.session import SessionType
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
async def test_swarm_interrupt_and_resume(researcher_agent, analyst_agent, writer_agent):
    """Test swarm interruption after analyst_agent and resume functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        session_id = str(uuid4())

        # Create session manager
        session_manager = FileSessionManager(
            session_id=session_id, storage_dir=temp_dir, session_type=SessionType.MULTI_AGENT
        )

        # Create swarm with session manager
        swarm = Swarm([researcher_agent, analyst_agent, writer_agent], session_manager=session_manager)

        # Mock analyst_agent to fail
        async def failing_invoke(*args, **kwargs):
            raise Exception("Simulated failure in analyst")

        with patch.object(analyst_agent, "invoke_async", side_effect=failing_invoke):
            # First execution - should fail at analyst
            result = await swarm.invoke_async("Research AI trends and create a brief report")
            assert result.status == Status.FAILED

        # Verify partial execution was persisted
        persisted_state = session_manager.read_multi_agent_json()
        assert persisted_state is not None
        assert persisted_state["type"] == "swarm"
        assert persisted_state["status"] == "failed"
        assert len(persisted_state["node_history"]) == 1  # At least researcher executed

        # Track execution count before resume
        initial_execution_count = len(persisted_state["node_history"])

        # Execute swarm again - should automatically resume from saved state
        result = await swarm.invoke_async("Research AI trends and create a brief report")

        # Verify successful completion
        assert result.status == Status.COMPLETED
        assert len(result.results) > 0

        assert len(result.node_history) >= initial_execution_count + 1

        node_names = [node.node_id for node in result.node_history]
        assert "researcher" in node_names
        # Either analyst or writer (or both) should have executed to complete the task
        assert "analyst" in node_names or "writer" in node_names

        # Clean up
        session_manager.delete_session(session_id)
