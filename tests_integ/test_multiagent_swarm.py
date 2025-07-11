import logging

import pytest

from strands import Agent, tool
from strands.multiagent.swarm import Swarm

logging.getLogger("strands.multiagent").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock implementation
    return f"Results for '{query}': Found information about {query}..."


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
        load_tools_from_directory=False,
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
        load_tools_from_directory=False,
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
        load_tools_from_directory=False,
    )


@pytest.mark.asyncio
async def test_swarm_execution(researcher_agent, analyst_agent, writer_agent):
    """Test basic swarm execution with multiple agents."""
    # Create the swarm
    swarm = Swarm([researcher_agent, analyst_agent, writer_agent])

    # Define a task that requires collaboration
    task = (
        "Research the current AI agent market trends, calculate the growth rate assuming 25% yearly growth, "
        "and create a basic report"
    )

    # Execute the swarm
    result = await swarm.execute_async(task)

    print(result)

    # Verify results
    assert result.status.value == "completed"
    assert len(result.results) > 0
    assert result.execution_time > 0
    assert result.execution_count > 0

    # Verify agent history - at least one agent should have been used
    assert len(result.node_history) > 0
