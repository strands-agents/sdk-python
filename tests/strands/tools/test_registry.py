"""
Tests for the SDK tool registry module.
"""

from unittest.mock import MagicMock

import pytest

import strands
from strands.tools import PythonAgentTool
from strands.tools.decorator import DecoratedFunctionTool, tool
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


def test_process_tools_flattens_lists_and_tuples_and_sets():
    def function() -> str:
        return "done"

    tool_a = tool(name="tool_a")(function)
    tool_b = tool(name="tool_b")(function)
    tool_c = tool(name="tool_c")(function)
    tool_d = tool(name="tool_d")(function)
    tool_e = tool(name="tool_e")(function)
    tool_f = tool(name="tool_f")(function)

    registry = ToolRegistry()

    all_tools = [tool_a, (tool_b, tool_c), [{tool_d, tool_e}, [tool_f]]]

    tru_tool_names = sorted(registry.process_tools(all_tools))
    exp_tool_names = [
        "tool_a",
        "tool_b",
        "tool_c",
        "tool_d",
        "tool_e",
        "tool_f",
    ]
    assert tru_tool_names == exp_tool_names


def test_register_tool_duplicate_name_without_hot_reload():
    """Test that registering a tool with duplicate name raises ValueError when hot reload is not supported."""
    tool_1 = PythonAgentTool(tool_name="duplicate_tool", tool_spec=MagicMock(), tool_func=lambda: None)
    tool_2 = PythonAgentTool(tool_name="duplicate_tool", tool_spec=MagicMock(), tool_func=lambda: None)

    tool_registry = ToolRegistry()
    tool_registry.register_tool(tool_1)

    with pytest.raises(
        ValueError, match="Tool name 'duplicate_tool' already exists. Cannot register tools with exact same name."
    ):
        tool_registry.register_tool(tool_2)


def test_register_tool_duplicate_name_with_hot_reload():
    """Test that registering a tool with duplicate name succeeds when hot reload is supported."""
    # Create mock tools with hot reload support
    tool_1 = MagicMock(spec=PythonAgentTool)
    tool_1.tool_name = "hot_reload_tool"
    tool_1.supports_hot_reload = True
    tool_1.is_dynamic = False

    tool_2 = MagicMock(spec=PythonAgentTool)
    tool_2.tool_name = "hot_reload_tool"
    tool_2.supports_hot_reload = True
    tool_2.is_dynamic = False

    tool_registry = ToolRegistry()
    tool_registry.register_tool(tool_1)

    tool_registry.register_tool(tool_2)

    # Verify the second tool replaced the first
    assert tool_registry.registry["hot_reload_tool"] == tool_2
