"""
Tests for the SDK callback handler module.

These tests ensure the basic print-based callback handler in the SDK functions correctly.
"""

import unittest.mock

import pytest

from strands.handlers.callback_handler import CompositeCallbackHandler, PrintingCallbackHandler


@pytest.fixture
def handler():
    """Create a fresh PrintingCallbackHandler instance for testing."""
    return PrintingCallbackHandler()


@pytest.fixture
def mock_print():
    with unittest.mock.patch("builtins.print") as mock:
        yield mock


def test_call_with_empty_args(handler, mock_print):
    """Test calling the handler with no arguments."""
    handler()
    # No output should be printed
    mock_print.assert_not_called()


def test_call_handler_reasoningText(handler, mock_print):
    """Test calling the handler with reasoningText."""
    handler(reasoningText="This is reasoning text")
    # Should print reasoning text without newline
    mock_print.assert_called_once_with("This is reasoning text", end="")


def test_call_without_reasoningText(handler, mock_print):
    """Test calling the handler without reasoningText argument."""
    handler(data="Some output")
    # Should only print data, not reasoningText
    mock_print.assert_called_once_with("Some output", end="")


def test_call_with_reasoningText_and_data(handler, mock_print):
    """Test calling the handler with both reasoningText and data."""
    handler(reasoningText="Reasoning", data="Output")
    # Should print reasoningText and data, both without newline
    calls = [
        unittest.mock.call("Reasoning", end=""),
        unittest.mock.call("Output", end=""),
    ]
    mock_print.assert_has_calls(calls)


def test_call_with_data_incomplete(handler, mock_print):
    """Test calling the handler with data but not complete."""
    handler(data="Test output")
    # Should print without newline
    mock_print.assert_called_once_with("Test output", end="")


def test_call_with_data_complete(handler, mock_print):
    """Test calling the handler with data and complete=True."""
    handler(data="Test output", complete=True)
    # Should print with newline
    # The handler prints the data, and then also prints a newline when complete=True and data exists
    assert mock_print.call_count == 2
    mock_print.assert_any_call("Test output", end="\n")
    mock_print.assert_any_call("\n")


def test_call_with_current_tool_use_new(handler, mock_print):
    """Test calling the handler with a new tool use."""
    current_tool_use = {"name": "test_tool", "input": {"param": "value"}}

    handler(current_tool_use=current_tool_use)

    # Should print tool information
    mock_print.assert_called_once_with("\nTool #1: test_tool")

    # Should update the handler state
    assert handler.tool_count == 1
    assert handler.previous_tool_use == current_tool_use


def test_call_with_current_tool_use_same(handler, mock_print):
    """Test calling the handler with the same tool use twice."""
    current_tool_use = {"name": "test_tool", "input": {"param": "value"}}

    # First call
    handler(current_tool_use=current_tool_use)
    mock_print.reset_mock()

    # Second call with same tool use
    handler(current_tool_use=current_tool_use)

    # Should not print tool information again
    mock_print.assert_not_called()

    # Tool count should not increase
    assert handler.tool_count == 1


def test_call_with_current_tool_use_different(handler, mock_print):
    """Test calling the handler with different tool uses."""
    first_tool_use = {"name": "first_tool", "input": {"param": "value1"}}
    second_tool_use = {"name": "second_tool", "input": {"param": "value2"}}

    # First call
    handler(current_tool_use=first_tool_use)
    mock_print.reset_mock()

    # Second call with different tool use
    handler(current_tool_use=second_tool_use)

    # Should print info for the new tool
    mock_print.assert_called_once_with("\nTool #2: second_tool")

    # Tool count should increase
    assert handler.tool_count == 2
    assert handler.previous_tool_use == second_tool_use


def test_call_with_data_and_complete_extra_newline(handler, mock_print):
    """Test that an extra newline is printed when data is complete."""
    handler(data="Test output", complete=True)

    # The handler prints the data with newline and an extra newline for completion
    assert mock_print.call_count == 2
    mock_print.assert_any_call("Test output", end="\n")
    mock_print.assert_any_call("\n")


def test_call_with_message_no_effect(handler, mock_print):
    """Test that passing a message without special content has no effect."""
    message = {"role": "user", "content": [{"text": "Hello"}]}

    handler(message=message)

    # No print calls should be made
    mock_print.assert_not_called()


def test_call_with_multiple_parameters(handler, mock_print):
    """Test calling handler with multiple parameters."""
    current_tool_use = {"name": "test_tool", "input": {"param": "value"}}

    handler(data="Test output", complete=True, current_tool_use=current_tool_use)

    # Should print data with newline, an extra newline for completion, and tool information
    assert mock_print.call_count == 3
    mock_print.assert_any_call("Test output", end="\n")
    mock_print.assert_any_call("\n")
    mock_print.assert_any_call("\nTool #1: test_tool")


def test_unknown_tool_name_handling(handler, mock_print):
    """Test handling of a tool use without a name."""
    # The SDK implementation doesn't have a fallback for tool uses without a name field
    # It checks for both presence of current_tool_use and current_tool_use.get("name")
    current_tool_use = {"input": {"param": "value"}, "name": "Unknown tool"}

    handler(current_tool_use=current_tool_use)

    # Should print the tool information
    mock_print.assert_called_once_with("\nTool #1: Unknown tool")


def test_tool_use_empty_object(handler, mock_print):
    """Test handling of an empty tool use object."""
    # Tool use is an empty dict
    current_tool_use = {}

    handler(current_tool_use=current_tool_use)

    # Should not print anything
    mock_print.assert_not_called()

    # Should not update state
    assert handler.tool_count == 0
    assert handler.previous_tool_use is None


def test_composite_handler_forwards_to_all_handlers():
    mock_handlers = [unittest.mock.Mock() for _ in range(3)]
    composite_handler = CompositeCallbackHandler(*mock_handlers)

    """Test that calling the handler forwards the call to all handlers."""
    # Create test arguments
    kwargs = {
        "data": "Test output",
        "complete": True,
        "current_tool_use": {"name": "test_tool", "input": {"param": "value"}},
    }

    # Call the composite handler
    composite_handler(**kwargs)

    # Verify each handler was called with the same arguments
    for handler in mock_handlers:
        handler.assert_called_once_with(**kwargs)
