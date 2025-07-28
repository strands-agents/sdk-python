#!/usr/bin/env python3
"""
Integration test for StrandsContext functionality with real agent interactions.
"""

import logging
from typing import Dict, Any

from strands import Agent, tool, StrandsContext

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


# Global storage to simulate persistent state across tool calls
GLOBAL_STORAGE: Dict[str, Any] = {}


@tool
def store_data(key: str, value: str, strands_context: StrandsContext) -> str:
    """
    Store data in global storage using tool_use_id for tracking.

    Args:
        key: The key to store the data under
        value: The value to store
    """
    tool_use_id = strands_context["tool_use"]["toolUseId"]
    tool_name = strands_context["tool_use"]["name"]
    
    # Store the data with metadata
    GLOBAL_STORAGE[key] = {
        "value": value,
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "timestamp": "2025-01-28T14:00:00Z"  # Simulated timestamp
    }
    
    return f"Stored '{value}' under key '{key}' (tracked with tool use ID: {tool_use_id})"


@tool
def retrieve_data(key: str, strands_context: StrandsContext) -> str:
    """
    Retrieve data from global storage with tracking information.

    Args:
        key: The key to retrieve data for
    """
    tool_use_id = strands_context["tool_use"]["toolUseId"]
    
    if key not in GLOBAL_STORAGE:
        return f"No data found for key '{key}' (retrieval tracked with ID: {tool_use_id})"
    
    stored_data = GLOBAL_STORAGE[key]
    return (f"Retrieved '{stored_data['value']}' for key '{key}'. "
            f"Originally stored by {stored_data['tool_name']} "
            f"(store ID: {stored_data['tool_use_id']}, retrieve ID: {tool_use_id})")


@tool
def list_storage(strands_context: StrandsContext) -> str:
    """
    List all stored data with tracking information.
    """
    tool_use_id = strands_context["tool_use"]["toolUseId"]
    
    if not GLOBAL_STORAGE:
        return f"Storage is empty (listed with tool use ID: {tool_use_id})"
    
    items = []
    for key, data in GLOBAL_STORAGE.items():
        items.append(f"  {key}: '{data['value']}' (stored by {data['tool_name']}, ID: {data['tool_use_id']})")
    
    return f"Storage contents (listed with tool use ID: {tool_use_id}):\n" + "\n".join(items)


@tool
def regular_tool_without_context(message: str) -> str:
    """
    A regular tool that doesn't use StrandsContext for comparison.

    Args:
        message: Message to process
    """
    return f"Regular tool processed: {message}"


def test_strands_context_integration():
    """Test StrandsContext functionality with real agent interactions."""
    
    print("\n===== StrandsContext Integration Test =====")
    
    # Clear global storage
    GLOBAL_STORAGE.clear()
    
    # Initialize agent with tools
    agent = Agent(tools=[store_data, retrieve_data, list_storage, regular_tool_without_context])
    
    print("\n1. Testing direct tool access with StrandsContext:")
    
    # Test storing data
    store_result = agent.tool.store_data(key="user_name", value="Alice")
    print(f"Store result: {store_result}")
    
    # Test retrieving data
    retrieve_result = agent.tool.retrieve_data(key="user_name")
    print(f"Retrieve result: {retrieve_result}")
    
    # Test listing storage
    list_result = agent.tool.list_storage()
    print(f"List result: {list_result}")
    
    # Test regular tool without context
    regular_result = agent.tool.regular_tool_without_context(message="Hello World")
    print(f"Regular tool result: {regular_result}")
    
    print("\n2. Testing natural language interactions:")
    
    # Store more data through natural language
    nl_store_result = agent("Store the value 'Bob' under the key 'friend_name'")
    print(f"NL Store result: {nl_store_result}")
    
    # Retrieve through natural language
    nl_retrieve_result = agent("What is stored under the key 'user_name'?")
    print(f"NL Retrieve result: {nl_retrieve_result}")
    
    # List all storage through natural language
    nl_list_result = agent("Show me everything in storage")
    print(f"NL List result: {nl_list_result}")
    
    print("\n3. Verifying global storage state:")
    print(f"Final storage state: {GLOBAL_STORAGE}")
    
    # Verify that tool_use_ids were properly tracked
    assert len(GLOBAL_STORAGE) == 2, f"Expected 2 items in storage, got {len(GLOBAL_STORAGE)}"
    assert "user_name" in GLOBAL_STORAGE, "user_name should be in storage"
    assert "friend_name" in GLOBAL_STORAGE, "friend_name should be in storage"
    
    # Verify metadata was stored
    user_data = GLOBAL_STORAGE["user_name"]
    assert "tool_use_id" in user_data, "tool_use_id should be tracked"
    assert "tool_name" in user_data, "tool_name should be tracked"
    assert user_data["tool_name"] == "store_data", "tool_name should be 'store_data'"
    
    print("\n✅ All StrandsContext integration tests passed!")


if __name__ == "__main__":
    test_strands_context_integration()
