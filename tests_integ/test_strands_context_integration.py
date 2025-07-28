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
    try:
        print(f"DEBUG: tool_with_context called with message='{message}'")
        print(f"DEBUG: strands_context type: {type(strands_context)}")
        print(f"DEBUG: strands_context contents: {strands_context}")

        tool_use_id = strands_context["tool_use"]["toolUseId"]
        print(f"DEBUG: Successfully extracted tool_use_id: {tool_use_id}")

        result = {
            "status": "success",
            "content": [{"text": f"Context tool processed '{message}' with ID: {tool_use_id}"}],
        }
        print(f"DEBUG: Returning result: {result}")
        return result

    except Exception as e:
        print(f"ERROR in tool_with_context: {type(e).__name__}: {e}")
        print(f"ERROR: strands_context = {strands_context}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


@tool
def tool_with_agent_and_context(message: str, agent: Agent, strands_context: StrandsContext) -> dict:
    """Tool that uses both agent and StrandsContext."""
    try:
        print(f"DEBUG: tool_with_agent_and_context called with message='{message}'")
        print(f"DEBUG: agent type: {type(agent)}")
        print(f"DEBUG: strands_context type: {type(strands_context)}")
        print(f"DEBUG: strands_context contents: {strands_context}")

        tool_use_id = strands_context["tool_use"]["toolUseId"]
        print(f"DEBUG: Successfully extracted tool_use_id: {tool_use_id}")

        agent_name = getattr(agent, "name", "unknown-agent")
        print(f"DEBUG: Agent name: {agent_name}")

        result = {
            "status": "success",
            "content": [{"text": f"Agent '{agent_name}' processed '{message}' with ID: {tool_use_id}"}],
        }
        print(f"DEBUG: Returning result: {result}")
        return result

    except Exception as e:
        print(f"ERROR in tool_with_agent_and_context: {type(e).__name__}: {e}")
        print(f"ERROR: agent = {agent}")
        print(f"ERROR: strands_context = {strands_context}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def test_strands_context_integration():
    """Test StrandsContext functionality with real agent interactions."""
    try:
        print("DEBUG: Starting test_strands_context_integration")

        # Initialize agent with tools
        print("DEBUG: Initializing agent with tools")
        agent = Agent(tools=[tool_with_context, tool_with_agent_and_context])
        print(f"DEBUG: Agent created: {agent}")

        # Test tool with StrandsContext
        print("DEBUG: Testing tool_with_context")
        result1 = agent.tool.tool_with_context(message="hello world")
        print(f"DEBUG: tool_with_context result: {result1}")
        assert result1.get("status") == "success"
        print("DEBUG: tool_with_context assertion passed")

        # Test tool with both agent and StrandsContext
        print("DEBUG: Testing tool_with_agent_and_context")
        result = agent.tool.tool_with_agent_and_context(message="hello agent")
        print(f"DEBUG: tool_with_agent_and_context result: {result}")
        assert result.get("status") == "success"
        print("DEBUG: tool_with_agent_and_context assertion passed")

        print("DEBUG: All tests passed successfully")

    except Exception as e:
        print(f"ERROR in test_strands_context_integration: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise
