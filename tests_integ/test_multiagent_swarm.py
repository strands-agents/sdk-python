import pytest

from strands import Agent, tool
from strands.multiagent.swarm import Swarm
from strands.types.content import ContentBlock


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
def researcher_agent():
    """Create an agent specialized in research."""
    return Agent(
        name="researcher",
        system_prompt=(
            "You are a research specialist who excels at finding information. When you need to perform calculations or"
            " format documents, hand off to the appropriate specialist."
        ),
        tools=[web_search],
    )


@pytest.fixture
def analyst_agent():
    """Create an agent specialized in data analysis."""
    return Agent(
        name="analyst",
        system_prompt=(
            "You are a data analyst who excels at calculations and numerical analysis. When you need"
            " research or document formatting, hand off to the appropriate specialist."
        ),
        tools=[calculate],
    )


@pytest.fixture
def writer_agent():
    """Create an agent specialized in writing and formatting."""
    return Agent(
        name="writer",
        system_prompt=(
            "You are a professional writer who excels at formatting and presenting information. When you need research"
            " or calculations, hand off to the appropriate specialist."
        ),
    )


def test_swarm_execution_with_string(researcher_agent, analyst_agent, writer_agent):
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
