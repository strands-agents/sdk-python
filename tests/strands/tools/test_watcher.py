"""
Tests for the SDK tool watcher module.
"""

from unittest.mock import MagicMock, patch

import pytest

from strands.tools.registry import ToolRegistry
from strands.tools.watcher import ToolWatcher


def test_tool_watcher_initialization():
    """Test that the handler initializes with the correct tool registry."""
    tool_registry = ToolRegistry()
    watcher = ToolWatcher(tool_registry)
    assert watcher.tool_registry == tool_registry


@pytest.mark.parametrize(
    "test_case",
    [
        # Regular Python file - should reload
        {
            "description": "Python file",
            "src_path": "/path/to/test_tool.py",
            "is_directory": False,
            "should_reload": True,
            "expected_tool_name": "test_tool",
        },
        # Non-Python file - should not reload
        {
            "description": "Non-Python file",
            "src_path": "/path/to/test_tool.txt",
            "is_directory": False,
            "should_reload": False,
        },
        # __init__.py file - should not reload
        {
            "description": "Init file",
            "src_path": "/path/to/__init__.py",
            "is_directory": False,
            "should_reload": False,
        },
        # Directory path - should not reload
        {
            "description": "Directory path",
            "src_path": "/path/to/tools_directory",
            "is_directory": True,
            "should_reload": False,
        },
        # Python file marked as directory - should still reload
        {
            "description": "Python file marked as directory",
            "src_path": "/path/to/test_tool2.py",
            "is_directory": True,
            "should_reload": True,
            "expected_tool_name": "test_tool2",
        },
    ],
)
@patch.object(ToolRegistry, "update_tool")
def test_on_modified_cases(mock_update_tool, test_case):
    """Test various cases for the on_modified method."""
    tool_registry = ToolRegistry()
    watcher = ToolWatcher(tool_registry)

    # Create a mock event with the specified properties
    event = MagicMock()
    event.src_path = test_case["src_path"]
    if "is_directory" in test_case:
        event.is_directory = test_case["is_directory"]

    # Call the on_modified method
    watcher.tool_change_handler.on_modified(event)

    # Verify the expected behavior
    if test_case["should_reload"]:
        mock_update_tool.assert_called_once_with(test_case["src_path"])
    else:
        mock_update_tool.assert_not_called()


@patch.object(ToolRegistry, "update_tool", side_effect=Exception("Test error"))
def test_on_modified_error_handling(mock_update_tool):
    """Test that on_modified handles errors during tool reloading."""
    tool_registry = ToolRegistry()
    watcher = ToolWatcher(tool_registry)

    # Create a mock event with a Python file path
    event = MagicMock()
    event.src_path = "/path/to/test_tool.py"

    # Call the on_modified method - should not raise an exception
    watcher.tool_change_handler.on_modified(event)

    # Verify that update_tool was called
    mock_update_tool.assert_called_once_with("/path/to/test_tool.py")
