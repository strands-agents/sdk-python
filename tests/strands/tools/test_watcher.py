"""
Tests for the SDK tool watcher module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands.tools.registry import ToolRegistry
from strands.tools.watcher import ToolWatcher


@pytest.fixture(autouse=True)
def reset_tool_watcher_state():
    """Reset ToolWatcher shared state between tests to avoid cross-test leakage."""
    ToolWatcher._shared_observer = None
    ToolWatcher._watched_dirs = set()
    ToolWatcher._observer_started = False
    ToolWatcher._registry_handlers = {}
    yield
    ToolWatcher._shared_observer = None
    ToolWatcher._watched_dirs = set()
    ToolWatcher._observer_started = False
    ToolWatcher._registry_handlers = {}


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


@patch("strands.tools.watcher.Observer")
def test_master_handler_routes_events_to_all_registries(mock_observer_cls):
    """Master handler should fan out file changes to all registry handlers for the same directory."""
    mock_observer = MagicMock()
    mock_observer_cls.return_value = mock_observer

    tools_dir = Path("/tmp/tools")
    registry_a = MagicMock(spec=ToolRegistry)
    registry_b = MagicMock(spec=ToolRegistry)
    registry_a.get_tools_dirs.return_value = [tools_dir]
    registry_b.get_tools_dirs.return_value = [tools_dir]

    ToolWatcher(registry_a)
    ToolWatcher(registry_b)

    # Only one observer/schedule/start for the shared directory
    mock_observer.schedule.assert_called_once()
    mock_observer.start.assert_called_once()
    assert len(ToolWatcher._registry_handlers[str(tools_dir)]) == 2

    event = MagicMock()
    event.src_path = str(tools_dir / "my_tool.py")
    event.is_directory = False

    master_handler = ToolWatcher.MasterChangeHandler(str(tools_dir))
    master_handler.on_modified(event)

    registry_a.reload_tool.assert_called_once_with("my_tool")
    registry_b.reload_tool.assert_called_once_with("my_tool")
