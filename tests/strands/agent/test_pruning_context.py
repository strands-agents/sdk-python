"""Tests for PruningContext class."""

from unittest.mock import Mock

import pytest

from strands.agent.conversation_manager import PruningContext
from strands.types.content import Messages


class TestPruningContext:
    """Test the PruningContext class."""

    @pytest.fixture
    def sample_messages(self) -> Messages:
        return [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {}}}]},
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "123", "content": [{"text": "Result"}], "status": "success"}}],
            },
            {"role": "assistant", "content": [{"text": "Final response"}]},
        ]

    @pytest.fixture
    def mock_agent(self):
        return Mock()

    def test_initialization(self, sample_messages, mock_agent):
        """Test PruningContext initialization."""
        context = PruningContext(sample_messages, mock_agent)

        assert context.messages == sample_messages
        assert context.agent == mock_agent
        assert len(context.token_counts) == len(sample_messages)

    def test_token_counts_estimation(self, sample_messages, mock_agent):
        """Test that token counts are estimated for each message."""
        context = PruningContext(sample_messages, mock_agent)

        assert len(context.token_counts) == len(sample_messages)
        assert all(isinstance(count, int) for count in context.token_counts)
        assert all(count >= 0 for count in context.token_counts)

    def test_get_message_context(self, sample_messages, mock_agent):
        """Test getting context for a specific message."""
        context = PruningContext(sample_messages, mock_agent)

        message_context = context.get_message_context(1)  # Assistant with tool use

        assert message_context["has_tool_use"] is True
        assert message_context["has_tool_result"] is False
        assert message_context["message_index"] == 1
        assert message_context["total_messages"] == 4
        assert isinstance(message_context["token_count"], int)

    def test_get_message_context_out_of_range(self, sample_messages, mock_agent):
        """Test that out of range indices raise IndexError."""
        context = PruningContext(sample_messages, mock_agent)

        with pytest.raises(IndexError, match="Message index 10 out of range"):
            context.get_message_context(10)

        with pytest.raises(IndexError, match="Message index -1 out of range"):
            context.get_message_context(-1)

    def test_tool_usage_detection(self, sample_messages, mock_agent):
        """Test tool usage detection in messages."""
        context = PruningContext(sample_messages, mock_agent)

        # Message 1 has tool use
        assert context._has_tool_use(sample_messages[1]) is True
        assert context._has_tool_result(sample_messages[1]) is False

        # Message 2 has tool result
        assert context._has_tool_use(sample_messages[2]) is False
        assert context._has_tool_result(sample_messages[2]) is True

        # Message 0 has neither
        assert context._has_tool_use(sample_messages[0]) is False
        assert context._has_tool_result(sample_messages[0]) is False

    def test_token_count_estimation_with_different_content_types(self, mock_agent):
        """Test token estimation with various content types."""
        messages: Messages = [
            {"role": "user", "content": [{"text": "Simple text message"}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {"param": "value"}}}],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool result text"}], "status": "success"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "456",
                            "content": [{"json": {"key": "value", "data": [1, 2, 3]}}],
                            "status": "success",
                        }
                    }
                ],
            },
        ]

        context = PruningContext(messages, mock_agent)

        # All messages should have positive token counts
        assert all(count > 0 for count in context.token_counts)

        # Tool use message should have reasonable token count (base + input)
        assert context.token_counts[1] >= 50  # Base tool use overhead

        # Tool result with JSON should have tokens estimated
        assert context.token_counts[3] > context.token_counts[0]  # JSON content should be more tokens

    def test_empty_messages_list(self, mock_agent):
        """Test behavior with empty messages list."""
        messages: Messages = []
        context = PruningContext(messages, mock_agent)

        assert context.token_counts == []

    def test_single_message(self, mock_agent):
        """Test behavior with single message."""
        messages: Messages = [{"role": "user", "content": [{"text": "Single message"}]}]
        context = PruningContext(messages, mock_agent)

        assert len(context.token_counts) == 1
        assert context.token_counts[0] > 0

        message_context = context.get_message_context(0)
        assert message_context["total_messages"] == 1

    def test_message_with_multiple_content_blocks(self, mock_agent):
        """Test token estimation for messages with multiple content blocks."""
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {"text": "First part of message"},
                    {"text": "Second part of message"},
                    {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}},
                ],
            }
        ]

        context = PruningContext(messages, mock_agent)

        # Should count tokens from all content blocks
        assert context.token_counts[0] > 0

        # Should detect tool result
        assert context._has_tool_result(messages[0]) is True
        assert context._has_tool_use(messages[0]) is False
