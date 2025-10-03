"""
Tests for the SDK tool watcher module.
"""

from pathlib import Path
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
@patch.object(ToolRegistry, "reload_tool")
def test_on_modified_cases(mock_reload_tool, test_case):
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
        mock_reload_tool.assert_called_once_with(test_case["expected_tool_name"])
    else:
        mock_reload_tool.assert_not_called()


@patch.object(ToolRegistry, "reload_tool", side_effect=Exception("Test error"))
def test_on_modified_error_handling(mock_reload_tool):
    """Test that on_modified handles errors during tool reloading."""
    tool_registry = ToolRegistry()
    watcher = ToolWatcher(tool_registry)

    # Create a mock event with a Python file path
    event = MagicMock()
    event.src_path = "/path/to/test_tool.py"

    # Call the on_modified method - should not raise an exception
    watcher.tool_change_handler.on_modified(event)

    # Verify that reload_tool was called
    mock_reload_tool.assert_called_once_with("test_tool")


@patch("strands.tools.watcher.ToolWatcher._shared_observer")
@patch.object(ToolRegistry, "get_tools_dirs")
def test_start_uses_correct_path_for_observer_schedule(mock_get_tools_dirs, mock_observer):
    """Test that start method schedules observer with correct path (str(tools_dir) not dir_str)."""

    # Mock the tools directories
    tools_dir = Path("/test/tools")
    mock_get_tools_dirs.return_value = [tools_dir]

    tool_registry = ToolRegistry()
    watcher = ToolWatcher(tool_registry)

    # Call start method
    watcher.start()

    # Verify that schedule was called with str(tools_dir)
    mock_observer.schedule.assert_called_once()
    args = mock_observer.schedule.call_args[0]
    assert args[1] == str(tools_dir)  # Second argument should be the string path
