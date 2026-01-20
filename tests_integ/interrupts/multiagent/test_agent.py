import json
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.interrupt import Interrupt
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.base import Status
from strands.types.tools import ToolContext


@pytest.fixture
def weather_tool():
    @tool(name="weather_tool", context=True)
    def func(tool_context: ToolContext) -> str:
        response = tool_context.interrupt("test_interrupt", reason="need weather")
        return response

    return func


@pytest.fixture
def swarm(weather_tool):
    weather_agent = Agent(name="weather", tools=[weather_tool])

    return Swarm([weather_agent])


def test_swarm_interrupt_agent(swarm):
    multiagent_result = swarm("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

    tru_interrupts = multiagent_result.interrupts
    exp_interrupts = [
        Interrupt(
            id=ANY,
            name="test_interrupt",
            reason="need weather",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    interrupt = multiagent_result.interrupts[0]

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "sunny",
            },
        },
    ]
    multiagent_result = swarm(responses)

    tru_status = multiagent_result.status
    exp_status = Status.COMPLETED
    assert tru_status == exp_status

    assert len(multiagent_result.results) == 1
    weather_result = multiagent_result.results["weather"]

    weather_message = json.dumps(weather_result.result.message).lower()
    assert "sunny" in weather_message


# ============================================================================
# Graph Agent Interrupt Integration Tests (Issue #1526)
# ============================================================================


@pytest.fixture
def graph(weather_tool):
    """Create a graph with an agent that can raise interrupts."""
    weather_agent = Agent(name="weather", tools=[weather_tool])

    builder = GraphBuilder()
    builder.add_node(weather_agent, "weather_agent")
    return builder.build()


def test_graph_interrupt_agent(graph):
    """Test that an agent node in a Graph can raise an interrupt and resume."""
    multiagent_result = graph("What is the weather?")

    tru_status = multiagent_result.status
    exp_status = Status.INTERRUPTED
    assert tru_status == exp_status

    tru_interrupts = multiagent_result.interrupts
    exp_interrupts = [
        Interrupt(
            id=ANY,
            name="test_interrupt",
            reason="need weather",
        ),
    ]
    assert tru_interrupts == exp_interrupts

    interrupt = multiagent_result.interrupts[0]

    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "sunny",
            },
        },
    ]
    multiagent_result = graph(responses)

    tru_status = multiagent_result.status
    exp_status = Status.COMPLETED
    assert tru_status == exp_status

    assert len(multiagent_result.results) == 1
    weather_result = multiagent_result.results["weather_agent"]

    weather_message = json.dumps(weather_result.result.message).lower()
    assert "sunny" in weather_message


def test_graph_interrupt_agent_parallel():
    """Test Graph with parallel agent nodes where one raises an interrupt."""

    @tool(name="interrupt_tool", context=True)
    def interrupt_tool(tool_context: ToolContext) -> str:
        response = tool_context.interrupt("approval", reason="need approval")
        return f"Approved: {response}"

    @tool(name="non_interrupt_tool")
    def non_interrupt_tool() -> str:
        return "Non-interrupt task completed"

    # Create two agents: one that will interrupt, one that won't
    interrupt_agent = Agent(name="interrupt_agent", tools=[interrupt_tool])
    non_interrupt_agent = Agent(name="non_interrupt_agent", tools=[non_interrupt_tool])

    builder = GraphBuilder()
    builder.add_node(interrupt_agent, "interrupt_agent")
    builder.add_node(non_interrupt_agent, "non_interrupt_agent")
    # Both are entry points, so they execute in parallel
    graph = builder.build()

    # First invocation - both agents start, interrupt_agent raises interrupt
    multiagent_result = graph("Execute tasks")

    assert multiagent_result.status == Status.INTERRUPTED
    assert len(multiagent_result.interrupts) == 1
    assert multiagent_result.interrupts[0].name == "approval"

    # non_interrupt_agent should have completed
    assert "completed_nodes" in graph._interrupt_state.context
    assert "non_interrupt_agent" in graph._interrupt_state.context["completed_nodes"]

    # Resume with response
    interrupt = multiagent_result.interrupts[0]
    responses = [
        {
            "interruptResponse": {
                "interruptId": interrupt.id,
                "response": "yes",
            },
        },
    ]
    multiagent_result = graph(responses)

    assert multiagent_result.status == Status.COMPLETED
    # Both agents should have results now
    assert len(multiagent_result.results) == 2
    assert "interrupt_agent" in multiagent_result.results
    assert "non_interrupt_agent" in multiagent_result.results
