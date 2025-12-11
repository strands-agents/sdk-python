"""Integration tests for FunctionNode with multiagent systems."""

import pytest

from strands import Agent
from strands.multiagent.base import Status
from strands.multiagent.function_node import FunctionNode
from strands.multiagent.graph import GraphBuilder

# Global variable to test function execution
test_global_var = None


def set_global_var(task, invocation_state=None, **kwargs):
    """Simple function that sets a global variable."""
    global test_global_var
    test_global_var = f"Function executed with: {task}"
    return "Global variable set"


@pytest.mark.asyncio
async def test_agent_with_function_node():
    """Test graph with agent and function node."""
    global test_global_var
    test_global_var = None

    # Create nodes
    agent = Agent()
    function_node = FunctionNode(set_global_var, "setter")

    # Build graph
    builder = GraphBuilder()
    builder.add_node(agent, "agent")
    builder.add_node(function_node, "setter")
    builder.add_edge("agent", "setter")
    builder.set_entry_point("agent")
    graph = builder.build()

    # Execute
    result = await graph.invoke_async("Say hello")

    # Verify function was called
    assert "Function executed with:" in test_global_var
    assert result.status == Status.COMPLETED
