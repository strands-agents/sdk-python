"""Unit tests for Bedrock-specific hooks."""

import unittest.mock
from unittest.mock import Mock

import pytest

from strands.hooks import HookRegistry
from strands.hooks.bedrock import CACHE_POINT_ITEM, PromptCachingHook
from strands.hooks.events import AfterModelCallEvent, BeforeModelCallEvent


@pytest.fixture
def hook():
    """Create a PromptCachingHook instance."""
    return PromptCachingHook()


@pytest.fixture
def mock_agent():
    """Create a mock agent with a messages list."""
    agent = Mock()
    agent.messages = []
    return agent


@pytest.fixture
def before_event(mock_agent):
    """Create a BeforeModelCallEvent with a mock agent."""
    return BeforeModelCallEvent(agent=mock_agent)


@pytest.fixture
def after_event(mock_agent):
    """Create an AfterModelCallEvent with a mock agent."""
    return AfterModelCallEvent(agent=mock_agent)


class TestPromptCachingHookRegistration:
    """Test hook registration functionality."""

    def test_register_hooks(self, hook):
        """Test that register_hooks adds callbacks to the registry."""
        registry = HookRegistry()
        hook.register_hooks(registry)

        # Verify callbacks are registered
        assert BeforeModelCallEvent in registry._registered_callbacks
        assert AfterModelCallEvent in registry._registered_callbacks
        assert len(registry._registered_callbacks[BeforeModelCallEvent]) == 1
        assert len(registry._registered_callbacks[AfterModelCallEvent]) == 1


class TestPromptCachingHookAddCachePoint:
    """Test adding cache points before model invocation."""

    def test_add_cache_point_success(self, hook, before_event):
        """Test successfully adding a cache point to the last message."""
        before_event.agent.messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
        ]

        hook.on_invocation_start(before_event)

        # Verify cache point was added
        assert len(before_event.agent.messages[-1]["content"]) == 2
        assert before_event.agent.messages[-1]["content"][-1] == CACHE_POINT_ITEM

    def test_add_cache_point_to_message_with_multiple_content_blocks(self, hook, before_event):
        """Test adding cache point to a message with multiple content blocks."""
        before_event.agent.messages = [
            {
                "role": "user",
                "content": [
                    {"text": "First block"},
                    {"text": "Second block"},
                    {"image": {"format": "png", "source": {"bytes": b"data"}}},
                ],
            },
        ]

        hook.on_invocation_start(before_event)

        # Verify cache point was added at the end
        assert len(before_event.agent.messages[-1]["content"]) == 4
        assert before_event.agent.messages[-1]["content"][-1] == CACHE_POINT_ITEM
        # Verify original content is intact
        assert before_event.agent.messages[-1]["content"][0] == {"text": "First block"}

    def test_add_cache_point_with_multiple_messages(self, hook, before_event):
        """Test that cache point is added only to the last message."""
        before_event.agent.messages = [
            {"role": "user", "content": [{"text": "First message"}]},
            {"role": "assistant", "content": [{"text": "Response"}]},
            {"role": "user", "content": [{"text": "Second message"}]},
        ]

        hook.on_invocation_start(before_event)

        # Verify cache point was added only to the last message
        assert len(before_event.agent.messages[0]["content"]) == 1
        assert len(before_event.agent.messages[1]["content"]) == 1
        assert len(before_event.agent.messages[2]["content"]) == 2
        assert before_event.agent.messages[2]["content"][-1] == CACHE_POINT_ITEM

    def test_add_cache_point_empty_messages_list(self, hook, before_event):
        """Test handling of empty messages list."""
        before_event.agent.messages = []

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_start(before_event)
            mock_logger.warning.assert_called_once_with("Cannot add cache point: messages list is empty")

        # Verify no error was raised and messages remain empty
        assert before_event.agent.messages == []

    def test_add_cache_point_message_without_content_field(self, hook, before_event):
        """Test handling of message without content field."""
        before_event.agent.messages = [
            {"role": "user"},  # No content field
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_start(before_event)
            mock_logger.warning.assert_called_once()
            assert "no content field" in mock_logger.warning.call_args[0][0]

        # Verify message was not modified
        assert "content" not in before_event.agent.messages[0]

    def test_add_cache_point_content_not_a_list(self, hook, before_event):
        """Test handling of content that is not a list."""
        before_event.agent.messages = [
            {"role": "user", "content": "This should be a list"},
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_start(before_event)
            mock_logger.warning.assert_called_once()
            assert "content is not a list" in mock_logger.warning.call_args[0][0]

        # Verify content was not modified
        assert before_event.agent.messages[0]["content"] == "This should be a list"


class TestPromptCachingHookRemoveCachePoint:
    """Test removing cache points after model invocation."""

    def test_remove_cache_point_success(self, hook, after_event):
        """Test successfully removing a cache point from the last message."""
        after_event.agent.messages = [
            {"role": "user", "content": [{"text": "Hello"}, CACHE_POINT_ITEM]},
        ]

        hook.on_invocation_end(after_event)

        # Verify cache point was removed
        assert len(after_event.agent.messages[-1]["content"]) == 1
        assert after_event.agent.messages[-1]["content"][0] == {"text": "Hello"}

    def test_remove_cache_point_from_message_with_multiple_blocks(self, hook, after_event):
        """Test removing cache point from a message with multiple content blocks."""
        after_event.agent.messages = [
            {
                "role": "user",
                "content": [
                    {"text": "First block"},
                    {"text": "Second block"},
                    CACHE_POINT_ITEM,
                ],
            },
        ]

        hook.on_invocation_end(after_event)

        # Verify only cache point was removed
        assert len(after_event.agent.messages[-1]["content"]) == 2
        assert after_event.agent.messages[-1]["content"][0] == {"text": "First block"}
        assert after_event.agent.messages[-1]["content"][1] == {"text": "Second block"}

    def test_remove_cache_point_with_multiple_messages(self, hook, after_event):
        """Test that cache point is removed only from the last message."""
        after_event.agent.messages = [
            {"role": "user", "content": [{"text": "First message"}, CACHE_POINT_ITEM]},
            {"role": "assistant", "content": [{"text": "Response"}]},
            {"role": "user", "content": [{"text": "Second message"}, CACHE_POINT_ITEM]},
        ]

        hook.on_invocation_end(after_event)

        # Verify cache point was removed only from the last message
        assert len(after_event.agent.messages[0]["content"]) == 2  # Unchanged
        assert len(after_event.agent.messages[1]["content"]) == 1  # No cache point
        assert len(after_event.agent.messages[2]["content"]) == 1  # Cache point removed

    def test_remove_cache_point_empty_messages_list(self, hook, after_event):
        """Test handling of empty messages list."""
        after_event.agent.messages = []

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_end(after_event)
            mock_logger.warning.assert_called_once_with("Cannot remove cache point: messages list is empty")

        # Verify no error was raised
        assert after_event.agent.messages == []

    def test_remove_cache_point_message_without_content_field(self, hook, after_event):
        """Test handling of message without content field."""
        after_event.agent.messages = [
            {"role": "user"},  # No content field
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_end(after_event)
            mock_logger.warning.assert_called_once()
            assert "no content field" in mock_logger.warning.call_args[0][0]

        # Verify message was not modified
        assert "content" not in after_event.agent.messages[0]

    def test_remove_cache_point_content_not_a_list(self, hook, after_event):
        """Test handling of content that is not a list."""
        after_event.agent.messages = [
            {"role": "user", "content": "This should be a list"},
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_end(after_event)
            mock_logger.warning.assert_called_once()
            assert "content is not a list" in mock_logger.warning.call_args[0][0]

        # Verify content was not modified
        assert after_event.agent.messages[0]["content"] == "This should be a list"

    def test_remove_cache_point_not_found(self, hook, after_event):
        """Test handling when cache point is not found in content."""
        after_event.agent.messages = [
            {"role": "user", "content": [{"text": "Hello"}]},  # No cache point
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            hook.on_invocation_end(after_event)
            mock_logger.warning.assert_called_once()
            assert "Cache point not found" in mock_logger.warning.call_args[0][0]

        # Verify content was not modified
        assert after_event.agent.messages[0]["content"] == [{"text": "Hello"}]


class TestPromptCachingHookEndToEnd:
    """Test end-to-end scenarios with both add and remove operations."""

    def test_add_and_remove_cache_point_lifecycle(self, hook, mock_agent):
        """Test the full lifecycle of adding and removing a cache point."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
        ]

        # Add cache point
        before_event = BeforeModelCallEvent(agent=mock_agent)
        hook.on_invocation_start(before_event)

        # Verify cache point was added
        assert len(mock_agent.messages[-1]["content"]) == 2
        assert mock_agent.messages[-1]["content"][-1] == CACHE_POINT_ITEM

        # Remove cache point
        after_event = AfterModelCallEvent(agent=mock_agent)
        hook.on_invocation_end(after_event)

        # Verify cache point was removed and original content is intact
        assert len(mock_agent.messages[-1]["content"]) == 1
        assert mock_agent.messages[-1]["content"][0] == {"text": "Hello"}

    def test_logging_on_successful_operations(self, hook, mock_agent):
        """Test that debug logs are generated on successful operations."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Test"}]},
        ]

        with unittest.mock.patch("strands.hooks.bedrock.logger") as mock_logger:
            # Add cache point
            before_event = BeforeModelCallEvent(agent=mock_agent)
            hook.on_invocation_start(before_event)

            # Verify debug log for adding
            mock_logger.debug.assert_called_once()
            assert "Added cache point" in mock_logger.debug.call_args[0][0]

            mock_logger.reset_mock()

            # Remove cache point
            after_event = AfterModelCallEvent(agent=mock_agent)
            hook.on_invocation_end(after_event)

            # Verify debug log for removing
            mock_logger.debug.assert_called_once()
            assert "Removed cache point" in mock_logger.debug.call_args[0][0]
