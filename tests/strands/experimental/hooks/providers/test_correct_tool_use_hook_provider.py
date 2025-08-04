"""Unit tests for CorrectToolUseHookProvider."""

from unittest.mock import Mock

import pytest

from strands.experimental.hooks.events import EventLoopFailureEvent
from strands.experimental.hooks.providers.correct_tool_use_hook_provider import CorrectToolUseHookProvider
from strands.hooks import HookRegistry
from strands.types.content import Message
from strands.types.exceptions import MaxTokensReachedException


@pytest.fixture
def hook_provider():
    """Create a CorrectToolUseHookProvider instance."""
    return CorrectToolUseHookProvider()


@pytest.fixture
def mock_agent():
    """Create a mock agent with messages and hooks."""
    agent = Mock()
    agent.messages = []
    agent.hooks = Mock()
    return agent


@pytest.fixture
def mock_registry():
    """Create a mock hook registry."""
    return Mock(spec=HookRegistry)


def test_register_hooks(hook_provider, mock_registry):
    """Test that the hook provider registers the correct callback."""
    hook_provider.register_hooks(mock_registry)

    mock_registry.add_callback.assert_called_once_with(EventLoopFailureEvent, hook_provider._handle_max_tokens_reached)


def test_handle_non_max_tokens_exception(hook_provider, mock_agent):
    """Test that non-MaxTokensReachedException events are ignored."""
    other_exception = ValueError("Some other error")
    event = EventLoopFailureEvent(agent=mock_agent, exception=other_exception)

    hook_provider._handle_max_tokens_reached(event)

    # Should not modify the agent or event
    assert len(mock_agent.messages) == 0
    assert not event.should_continue_loop
    mock_agent.hooks.invoke_callbacks.assert_not_called()


@pytest.mark.parametrize(
    "incomplete_tool_use,expected_tool_name",
    [
        ({"toolUseId": "tool-123", "input": {"param": "value"}}, "<unknown>"),  # Missing name
        ({"name": "test_tool", "toolUseId": "tool-123"}, "test_tool"),  # Missing input
        ({"name": "test_tool", "input": {}, "toolUseId": "tool-123"}, "test_tool"),  # Empty input
        ({"name": "test_tool", "input": {"param": "value"}}, "test_tool"),  # Missing toolUseId
    ],
)
def test_handle_max_tokens_with_incomplete_tool_use(hook_provider, mock_agent, incomplete_tool_use, expected_tool_name):
    """Test handling various incomplete tool use scenarios."""
    incomplete_message: Message = {
        "role": "user",  # Test role preservation
        "content": [{"text": "I'll use a tool"}, {"toolUse": incomplete_tool_use}],
    }

    exception = MaxTokensReachedException("Max tokens reached", incomplete_message)
    event = EventLoopFailureEvent(agent=mock_agent, exception=exception)

    hook_provider._handle_max_tokens_reached(event)

    # Should add corrected message with error text and preserve role
    assert len(mock_agent.messages) == 1
    added_message = mock_agent.messages[0]
    assert added_message["role"] == "user"  # Role preserved
    assert len(added_message["content"]) == 2
    assert added_message["content"][0]["text"] == "I'll use a tool"
    assert f"The selected tool {expected_tool_name}'s tool use was incomplete" in added_message["content"][1]["text"]
    assert "maximum token limits being reached" in added_message["content"][1]["text"]

    assert event.should_continue_loop


def test_handle_max_tokens_with_no_content(hook_provider, mock_agent):
    """Test handling message with no content blocks."""
    incomplete_message: Message = {"role": "assistant", "content": []}

    exception = MaxTokensReachedException("Max tokens reached", incomplete_message)
    event = EventLoopFailureEvent(agent=mock_agent, exception=exception)

    hook_provider._handle_max_tokens_reached(event)

    # Should add empty message and continue
    assert len(mock_agent.messages) == 0
    assert not event.should_continue_loop
