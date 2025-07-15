"""
Integration tests for the ToolRegistry CRUDL methods.

These tests verify that the Create, Read, Update, Delete, and List methods
of the ToolRegistry class work correctly in a real-world scenario.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from strands import Agent, tool
from strands.tools.registry import ToolRegistry


def test_create_tool():
    """Test creating a tool using the create_tool method."""
    # Create a tool registry
    registry = ToolRegistry()

    # Define a tool function
    @tool
    def calculator_add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    # Create the tool in the registry
    tool_name = registry.create_tool(calculator_add)

    # Verify the tool was created
    assert tool_name == "calculator_add"
    assert "calculator_add" in registry.agent_tools
    assert registry.agent_tools["calculator_add"].tool_spec["name"] == "calculator_add"
    assert registry.agent_tools["calculator_add"].tool_spec["description"] == "Add two numbers together."


def test_read_tool():
    """Test reading a tool using the read_tool method."""
    # Create a tool registry with a tool
    registry = ToolRegistry()

    @tool
    def calculator_subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    registry.create_tool(calculator_subtract)

    # Read the tool
    tool_spec = registry.read_tool("calculator_subtract")

    # Verify the tool spec
    assert tool_spec["name"] == "calculator_subtract"
    assert tool_spec["description"] == "Subtract b from a."
    assert "inputSchema" in tool_spec
    assert tool_spec["inputSchema"]["json"]["properties"]["a"]["type"] == "integer"
    assert tool_spec["inputSchema"]["json"]["properties"]["b"]["type"] == "integer"


def test_update_tool():
    """Test updating a tool using the update_tool method."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create a file with a predictable name
    tool_path = os.path.join(temp_dir, "calculator_multiply.py")
    with open(tool_path, "w") as f:
        f.write('''
from strands import tool

@tool
def calculator_multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
''')

    try:
        # Create a tool registry and add the tool
        registry = ToolRegistry()
        registry.create_tool(tool_path)

        # Verify the tool was created
        assert "calculator_multiply" in registry.agent_tools
        assert registry.agent_tools["calculator_multiply"].tool_spec["description"] == "Multiply two numbers."

        # Update the tool file
        with open(tool_path, "w") as f:
            f.write('''
from strands import tool

@tool
def calculator_multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b
''')

        # Update the tool in the registry using the absolute path
        updated_tool = registry.update_tool(os.path.abspath(tool_path))

        # Verify the tool was updated
        assert updated_tool == "calculator_multiply"
        assert (
            registry.agent_tools["calculator_multiply"].tool_spec["description"]
            == "Multiply two numbers together and return the result."
        )

    finally:
        # Clean up the temporary directory
        import shutil

        shutil.rmtree(temp_dir)


def test_delete_tool():
    """Test deleting a tool using the delete_tool method."""
    # Create a tool registry with multiple tools
    registry = ToolRegistry()

    @tool
    def calculator_add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool
    def calculator_subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    registry.create_tool(calculator_add)
    registry.create_tool(calculator_subtract)

    # Verify both tools exist
    assert "calculator_add" in registry.agent_tools
    assert "calculator_subtract" in registry.agent_tools

    # Delete one tool
    deleted_tool = registry.delete_tool("calculator_add")

    # Verify the tool was deleted
    assert deleted_tool == "calculator_add"
    assert "calculator_add" not in registry.agent_tools
    assert "calculator_subtract" in registry.agent_tools

    # Try to delete a non-existent tool
    with pytest.raises(ValueError):
        registry.delete_tool("non_existent_tool")


def test_list_tools():
    """Test listing all tools using the list_tools method."""
    # Create a tool registry with multiple tools
    registry = ToolRegistry()

    @tool
    def calculator_add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool
    def calculator_subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    @tool
    def calculator_multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    registry.create_tool(calculator_add)
    registry.create_tool(calculator_subtract)
    registry.create_tool(calculator_multiply)

    # List all tools
    tools = registry.list_tools()

    # Verify the tools were listed
    assert len(tools) == 3
    assert "calculator_add" in tools
    assert "calculator_subtract" in tools
    assert "calculator_multiply" in tools
    assert tools["calculator_add"]["name"] == "calculator_add"
    assert tools["calculator_subtract"]["name"] == "calculator_subtract"
    assert tools["calculator_multiply"]["name"] == "calculator_multiply"


def test_tool_registry_with_agent():
    """Test using the ToolRegistry with an Agent."""
    # Create a tool registry with a tool
    registry = ToolRegistry()

    @tool
    def calculator_add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    registry.create_tool(calculator_add)

    # Create an agent with the registry
    agent = Agent(load_tools_from_directory=False)
    agent.tool_registry = registry

    # Use the tool through the agent
    result = agent.tool.calculator_add(a=5, b=3)

    # Verify the result
    assert result["status"] == "success"
    assert result["content"][0]["text"] == "8"


def test_load_tools_from_directory():
    """Test the ToolRegistry with load_tools_from_directory=True."""
    # Create a tools directory in the current working directory
    tools_dir = Path.cwd() / "tools"
    os.makedirs(tools_dir, exist_ok=True)

    # Create a tool file in the tools directory
    tool_path = tools_dir / "directory_tool.py"
    with open(tool_path, "w") as f:
        f.write('''
from strands import tool

@tool
def directory_calculator(a: int, b: int) -> int:
    """Calculate the sum of two numbers from a directory-loaded tool."""
    return a + b
''')

    try:
        # Create a registry with load_tools_from_directory=True
        registry = ToolRegistry(load_tools_from_directory=True)

        # Wait a moment for the tools to be loaded
        time.sleep(2)

        # Verify the tool was loaded
        assert "directory_calculator" in registry.agent_tools

        # Read the tool spec
        tool_spec = registry.read_tool("directory_calculator")
        assert tool_spec["name"] == "directory_calculator"
        assert tool_spec["description"] == "Calculate the sum of two numbers from a directory-loaded tool."

        # Create a new agent with load_tools_from_directory=True
        agent = Agent(load_tools_from_directory=True)

        # Wait for the agent to load tools
        time.sleep(2)

        # Verify the tool is available in the agent
        assert "directory_calculator" in agent.tool_names

        # Use the tool
        result = agent.tool.directory_calculator(a=10, b=5)
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "15"

    finally:
        # Clean up
        if os.path.exists(tool_path):
            os.remove(tool_path)
