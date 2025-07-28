#!/usr/bin/env python3
"""
Integration test for StrandsContext functionality with real agent interactions.
"""

import logging

from strands import Agent, StrandsContext, tool

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


@tool
def tool_with_context(message: str, strands_context: StrandsContext) -> dict:
    """Tool that uses StrandsContext to access tool_use_id."""
    tool_use_id = strands_context["tool_use"]["toolUseId"]
    return {"status": "success", "content": [{"text": f"Context tool processed '{message}' with ID: {tool_use_id}"}]}


@tool
def tool_with_agent_and_context(message: str, agent: Agent, strands_context: StrandsContext) -> dict:
    """Tool that uses both agent and StrandsContext."""
    tool_use_id = strands_context["tool_use"]["toolUseId"]
    agent_name = getattr(agent, "name", "unknown-agent")
    return {
        "status": "success",
        "content": [{"text": f"Agent '{agent_name}' processed '{message}' with ID: {tool_use_id}"}],
    }


def test_strands_context_integration():
    """Test StrandsContext functionality with real agent interactions."""

    # Initialize agent with tools
    agent = Agent(tools=[tool_with_context, tool_with_agent_and_context])

    # Test tool with StrandsContext
    result1 = agent.tool.tool_with_context(message="hello world")
    assert result1.get("status") == "success"

    # Test tool with both agent and StrandsContext
    result = agent.tool.tool_with_agent_and_context(message="hello agent")
    assert result.get("status") == "success"
