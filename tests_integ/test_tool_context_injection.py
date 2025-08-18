#!/usr/bin/env python3
"""
Integration test for ToolContext functionality with real agent interactions.
"""

from strands import Agent, ToolContext, tool
from strands.types.tools import ToolResult


@tool(context="custom_context_field")
def good_story(message: str, custom_context_field: ToolContext) -> dict:
    """Tool that writes a good story"""
    tool_use_id = custom_context_field.tool_use["toolUseId"]
    return {
        "status": "success",
        "content": [{"text": f"Context tool processed with ID: {tool_use_id}"}],
    }


@tool(context=True)
def bad_story(message: str, tool_context: ToolContext) -> dict:
    """Tool that writes a bad story"""
    tool_use_id = tool_context.tool_use["toolUseId"]
    return {
        "status": "success",
        "content": [{"text": f"Context tool processed with ID: {tool_use_id}"}],
    }


def _validate_tool_result_content(agent: Agent):
    first_tool_result: ToolResult = [
        block["toolResult"] for message in agent.messages for block in message["content"] if "toolResult" in block
    ][0]

    assert first_tool_result["status"] == "success"
    assert (
        first_tool_result["content"][0]["text"] == f"Context tool processed with ID: {first_tool_result['toolUseId']}"
    )


def test_strands_context_integration_context_true():
    """Test ToolContext functionality with real agent interactions."""

    agent = Agent(tools=[good_story])
    agent("using a tool, write a good story")

    _validate_tool_result_content(agent)


def test_strands_context_integration_context_custom():
    """Test ToolContext functionality with real agent interactions."""

    agent = Agent(tools=[bad_story])
    agent("using a tool, write a bad story")

    _validate_tool_result_content(agent)
