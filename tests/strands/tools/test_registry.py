"""
Tests for the SDK tool registry module.
"""

from unittest.mock import MagicMock

import pytest

import strands
from strands.agent.agent import Agent
from strands.tools import PythonAgentTool
from strands.tools.agent_tool_wrapper import AgentToolWrapper
from strands.tools.decorator import DecoratedFunctionTool, tool
from strands.tools.registry import ToolRegistry
from tests.fixtures.mocked_model_provider import MockedModelProvider


def test_load_tool_from_filepath_failure():
    """Test error handling when load_tool fails."""
    tool_registry = ToolRegistry()
    error_message = "Failed to load tool failing_tool: Tool file not found: /path/to/failing_tool.py"

    with pytest.raises(ValueError, match=error_message):
        tool_registry.load_tool_from_filepath("failing_tool", "/path/to/failing_tool.py")


def test_process_tools_with_invalid_path():
    """Test that process_tools raises an exception when a non-path string is passed."""
    tool_registry = ToolRegistry()
    invalid_path = "not a filepath"

    with pytest.raises(ValueError, match=f"Failed to load tool {invalid_path.split('.')[0]}: Tool file not found:.*"):
        tool_registry.process_tools([invalid_path])


def test_register_tool_with_similar_name_raises():
    tool_1 = PythonAgentTool(tool_name="tool-like-this", tool_spec=MagicMock(), tool_func=lambda: None)
    tool_2 = PythonAgentTool(tool_name="tool_like_this", tool_spec=MagicMock(), tool_func=lambda: None)

    tool_registry = ToolRegistry()

    tool_registry.register_tool(tool_1)

    with pytest.raises(ValueError) as err:
        tool_registry.register_tool(tool_2)

    assert (
        str(err.value) == "Tool name 'tool_like_this' already exists as 'tool-like-this'. "
        "Cannot add a duplicate tool which differs by a '-' or '_'"
    )


def test_get_all_tool_specs_returns_right_tool_specs():
    tool_1 = strands.tool(lambda a: a, name="tool_1")
    tool_2 = strands.tool(lambda b: b, name="tool_2")

    tool_registry = ToolRegistry()

    tool_registry.register_tool(tool_1)
    tool_registry.register_tool(tool_2)

    tool_specs = tool_registry.get_all_tool_specs()

    assert tool_specs == [
        tool_1.tool_spec,
        tool_2.tool_spec,
    ]


def test_scan_module_for_tools():
    @tool
    def tool_function_1(a):
        return a

    @tool
    def tool_function_2(b):
        return b

    def tool_function_3(c):
        return c

    def tool_function_4(d):
        return d

    tool_function_4.tool_spec = "invalid"

    mock_module = MagicMock()
    mock_module.tool_function_1 = tool_function_1
    mock_module.tool_function_2 = tool_function_2
    mock_module.tool_function_3 = tool_function_3
    mock_module.tool_function_4 = tool_function_4

    tool_registry = ToolRegistry()

    tools = tool_registry._scan_module_for_tools(mock_module)

    assert len(tools) == 2
    assert all(isinstance(tool, DecoratedFunctionTool) for tool in tools)


def test_process_tools_with_agent():
    """Test that process_tools correctly wraps Agent objects with AgentToolWrapper."""

    # Create a real Agent instance using the mocked model provider
    model = MockedModelProvider([])
    agent = Agent(model=model, name="test_agent", description="A test agent for testing")

    tool_registry = ToolRegistry()

    # Process the agent as a tool
    tool_names = tool_registry.process_tools([agent])

    # Verify the agent was processed
    assert len(tool_names) == 1
    assert tool_names[0] == "test_agent"

    # Verify the agent was wrapped and registered
    assert "test_agent" in tool_registry.registry
    wrapped_tool = tool_registry.registry["test_agent"]
    assert isinstance(wrapped_tool, AgentToolWrapper)
    assert wrapped_tool.tool_name == "test_agent"
    assert wrapped_tool.tool_type == "agent"


def test_process_tools_with_non_agent_objects():
    """Test that process_tools correctly handles non-Agent objects and doesn't register them."""
    tool_registry = ToolRegistry()

    # Create a non-Agent object that might look like an agent
    class NotAnAgent:
        def __init__(self):
            self.name = "test_name"
            self.description = "A description"

    not_agent = NotAnAgent()

    # Process the non-agent object
    tool_names = tool_registry.process_tools([not_agent])

    # Verify the non-agent was not processed or registered
    assert len(tool_names) == 0
    assert "test_name" not in tool_registry.registry

    # Test with other non-agent objects
    tool_names = tool_registry.process_tools([123, None, {"not_a_tool": "spec"}])
    assert len(tool_names) == 0


def test_is_agent_instance():
    """Test the _is_agent_instance method correctly identifies agent objects."""
    tool_registry = ToolRegistry()

    # Create a real Agent instance using the mocked model provider
    model = MockedModelProvider([])
    agent = Agent(model=model, name="test_agent", description="A test agent")

    assert tool_registry._is_agent_instance(agent) is True

    # Test with non-Agent objects
    class NotAnAgent:
        def __init__(self):
            self.name = "test_name"
            self.description = "A description"

    not_agent = NotAnAgent()
    assert tool_registry._is_agent_instance(not_agent) is False

    # Test with regular object
    assert tool_registry._is_agent_instance("not an agent") is False

    # Test with None
    assert tool_registry._is_agent_instance(None) is False
