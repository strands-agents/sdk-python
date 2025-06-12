"""
Tests for the SDK tool registry module.
"""

from unittest.mock import MagicMock

import pytest

from strands import tool
from strands.tools import PythonAgentTool
from strands.tools.registry import ToolRegistry


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
    tool_1 = PythonAgentTool(tool_name="tool-like-this", tool_spec=MagicMock(), callback=lambda: None)
    tool_2 = PythonAgentTool(tool_name="tool_like_this", tool_spec=MagicMock(), callback=lambda: None)

    tool_registry = ToolRegistry()

    tool_registry.register_tool(tool_1)

    with pytest.raises(ValueError) as err:
        tool_registry.register_tool(tool_2)

    assert (
        str(err.value) == "Tool name 'tool_like_this' already exists as 'tool-like-this'. "
        "Cannot add a duplicate tool which differs by a '-' or '_'"
    )


@pytest.fixture
def test_agent():
    class TestAgent:
        @tool
        def word_counter(self, text: str) -> str:
            """Count words in text"""
            return f"Word count: {len(text.split())}"

    return TestAgent()


@pytest.fixture
def tool_registry():
    return ToolRegistry()


def test_class_method_register_tool(test_agent, tool_registry):
    registered_tool_names = tool_registry.process_tools([test_agent.word_counter])
    assert len(registered_tool_names) == 1
    assert registered_tool_names[0] == "word_counter"
