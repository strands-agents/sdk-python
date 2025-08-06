"""Tests for token limit recovery utility."""

from unittest.mock import Mock

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.recover_tool_use_on_max_tokens_reached import (
    recover_tool_use_on_max_tokens_reached,
)
from strands.hooks import MessageAddedEvent
from strands.types.content import Message
from strands.types.exceptions import MaxTokensReachedException


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_with_incomplete_tool_use():
    """Test recovery when incomplete tool use is present in the message."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"text": "I'll help you with that."},
            {"toolUse": {"name": "calculator", "input": {}, "toolUseId": ""}},  # Missing toolUseId
        ],
    }

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should add one corrected message
    assert len(agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 2

    # First content block should be preserved
    assert corrected_message["content"][0] == {"text": "I'll help you with that."}

    # Second content block should be replaced with error message
    assert "text" in corrected_message["content"][1]
    assert "calculator" in corrected_message["content"][1]["text"]
    assert "incomplete due to maximum token limits" in corrected_message["content"][1]["text"]

    # Verify that the MessageAddedEvent callback was invoked
    mock_invoke_callbacks.assert_called_once()
    call_args = mock_invoke_callbacks.call_args[0][0]
    assert isinstance(call_args, MessageAddedEvent)
    assert call_args.agent == agent
    assert call_args.message == corrected_message


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_with_unknown_tool_name():
    """Test recovery when tool use has no name."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"name": "", "input": {}, "toolUseId": "123"}},  # Missing name
        ],
    }

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should add one corrected message
    assert len(agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 1

    # Content should be replaced with error message using <unknown>
    assert "text" in corrected_message["content"][0]
    assert "<unknown>" in corrected_message["content"][0]["text"]
    assert "incomplete due to maximum token limits" in corrected_message["content"][0]["text"]

    # Verify that the MessageAddedEvent callback was invoked
    mock_invoke_callbacks.assert_called_once()
    call_args = mock_invoke_callbacks.call_args[0][0]
    assert isinstance(call_args, MessageAddedEvent)
    assert call_args.agent == agent
    assert call_args.message == corrected_message


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_with_valid_tool_use():
    """Test that an exception that is raised without recoverability, re-raises exception."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"text": "I'll help you with that."},
            {"toolUse": {"name": "calculator", "input": {"expression": "2+2"}, "toolUseId": "123"}},  # Valid
        ],
    }

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    with pytest.raises(MaxTokensReachedException):
        await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should not add any message since tool use was valid
    assert len(agent.messages) == initial_message_count

    # Verify that the MessageAddedEvent callback was NOT invoked
    mock_invoke_callbacks.assert_not_called()


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_with_empty_content():
    """Test that an exception that is raised without recoverability, re-raises exception."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {"role": "assistant", "content": []}

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    with pytest.raises(MaxTokensReachedException):
        await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should not add any message since content is empty
    assert len(agent.messages) == initial_message_count

    # Verify that the MessageAddedEvent callback was NOT invoked
    mock_invoke_callbacks.assert_not_called()


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_with_mixed_content():
    """Test recovery with mix of valid content and incomplete tool use."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"text": "Let me calculate this for you."},
            {"toolUse": {"name": "calculator", "input": {}, "toolUseId": ""}},  # Incomplete
            {"text": "And then I'll explain the result."},
        ],
    }

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should add one corrected message
    assert len(agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 3

    # First and third content blocks should be preserved
    assert corrected_message["content"][0] == {"text": "Let me calculate this for you."}
    assert corrected_message["content"][2] == {"text": "And then I'll explain the result."}

    # Second content block should be replaced with error message
    assert "text" in corrected_message["content"][1]
    assert "calculator" in corrected_message["content"][1]["text"]
    assert "incomplete due to maximum token limits" in corrected_message["content"][1]["text"]

    # Verify that the MessageAddedEvent callback was invoked
    mock_invoke_callbacks.assert_called_once()
    call_args = mock_invoke_callbacks.call_args[0][0]
    assert isinstance(call_args, MessageAddedEvent)
    assert call_args.agent == agent
    assert call_args.message == corrected_message


@pytest.mark.asyncio
async def test_recover_tool_use_on_max_tokens_reached_preserves_non_tool_content():
    """Test that non-tool content is preserved as-is."""
    agent = Agent()
    # Mock the hooks.invoke_callbacks method
    mock_invoke_callbacks = Mock()
    agent.hooks.invoke_callbacks = mock_invoke_callbacks
    initial_message_count = len(agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"text": "Here's some text."},
            {"image": {"format": "png", "source": {"bytes": "fake_image_data"}}},
            {"toolUse": {"name": "", "input": {}, "toolUseId": "123"}},  # Incomplete
        ],
    }

    exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await recover_tool_use_on_max_tokens_reached(agent, exception)

    # Should add one corrected message
    assert len(agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 3

    # First two content blocks should be preserved exactly
    assert corrected_message["content"][0] == {"text": "Here's some text."}
    assert corrected_message["content"][1] == {"image": {"format": "png", "source": {"bytes": "fake_image_data"}}}

    # Third content block should be replaced with error message
    assert "text" in corrected_message["content"][2]
    assert "<unknown>" in corrected_message["content"][2]["text"]

    # Verify that the MessageAddedEvent callback was invoked
    mock_invoke_callbacks.assert_called_once()
    call_args = mock_invoke_callbacks.call_args[0][0]
    assert isinstance(call_args, MessageAddedEvent)
    assert call_args.agent == agent
    assert call_args.message == corrected_message
