"""Tests for MappingConversationManager and message mappers."""

from typing import Optional
from unittest.mock import Mock

import pytest

from strands.experimental.conversation_manager.mapping_conversation_manager import (
    LargeToolResultMapper,
    MappingConversationManager,
)
from strands.types.content import Message, Messages


def simple_remove_old_mapper(message: Message, index: int, messages: Messages) -> Optional[Message]:
    """Test mapper that removes messages containing 'Old message'."""
    for content in message.get("content", []):
        if "text" in content and "Old message" in content["text"]:
            return None
    return message


class TestMappingConversationManager:
    """Test the MappingConversationManager class."""

    @pytest.fixture
    def large_result_mapper(self):
        return LargeToolResultMapper(max_tokens=100)

    @pytest.fixture
    def manager(self, large_result_mapper):
        return MappingConversationManager(
            mapper=large_result_mapper,
            preserve_first=1,
            preserve_last=2,
        )

    @pytest.fixture
    def simple_manager(self):
        """Manager with simple test mapper."""
        return MappingConversationManager(
            mapper=simple_remove_old_mapper,
            preserve_first=1,
            preserve_last=2,
        )

    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.messages = []
        return agent

    def test_initialization(self, large_result_mapper):
        """Test manager initialization with parameters."""
        manager = MappingConversationManager(
            mapper=large_result_mapper,
            preserve_first=2,
            preserve_last=5,
        )

        assert manager.mapper == large_result_mapper
        assert manager.preserve_first == 2
        assert manager.preserve_last == 5
        assert manager.removed_message_count == 0

    def test_reduce_context_with_empty_messages(self, manager, mock_agent):
        """Test that reduce_context returns early with empty messages."""
        mock_agent.messages = []

        # Should return early without raising exception
        manager.reduce_context(mock_agent)
        assert len(mock_agent.messages) == 0
        assert manager.removed_message_count == 0

    def test_reduce_context_with_insufficient_messages(self, manager, mock_agent):
        """Test behavior with too few messages to map safely."""
        mock_agent.messages = [{"role": "user", "content": [{"text": "Message 1"}]}]

        # Should return early without raising exception
        manager.reduce_context(mock_agent)
        assert len(mock_agent.messages) == 1
        assert manager.removed_message_count == 0

    def test_successful_mapping_removes_messages(self, simple_manager, mock_agent):
        """Test successful message removal via mapping."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial"}]},  # Preserved
            {"role": "user", "content": [{"text": "Old message 1"}]},  # Removed
            {"role": "user", "content": [{"text": "Old message 2"}]},  # Removed
            {"role": "user", "content": [{"text": "Recent 1"}]},  # Preserved
            {"role": "user", "content": [{"text": "Recent 2"}]},  # Preserved
        ]

        original_count = len(mock_agent.messages)
        simple_manager.reduce_context(mock_agent)

        # Should have removed the "Old message" entries
        assert len(mock_agent.messages) < original_count
        assert simple_manager.removed_message_count == 2

        # Check preserved messages remain
        assert mock_agent.messages[0]["content"][0]["text"] == "Initial"
        assert "Recent" in mock_agent.messages[-1]["content"][0]["text"]

    def test_preserve_first_messages(self, simple_manager, mock_agent):
        """Test that first messages are never mapped."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Old message initial"}]},  # Should be preserved despite "Old"
            {"role": "user", "content": [{"text": "Old message 2"}]},  # Should be removed
            {"role": "user", "content": [{"text": "Recent 1"}]},
            {"role": "user", "content": [{"text": "Recent 2"}]},
        ]

        simple_manager.reduce_context(mock_agent)

        # First message should still be there even though it contains "Old message"
        assert mock_agent.messages[0]["content"][0]["text"] == "Old message initial"

    def test_preserve_last_messages(self, simple_manager, mock_agent):
        """Test that last messages are never mapped."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial"}]},
            {"role": "user", "content": [{"text": "Old message"}]},  # Should be removed
            {"role": "user", "content": [{"text": "Old message recent 1"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Old message recent 2"}]},  # Should be preserved
        ]

        simple_manager.reduce_context(mock_agent)

        # Last two messages should be preserved
        assert len(mock_agent.messages) >= 2
        assert "Old message recent" in mock_agent.messages[-1]["content"][0]["text"]
        assert "Old message recent" in mock_agent.messages[-2]["content"][0]["text"]

    def test_lambda_mapper(self, mock_agent):
        """Test using lambda functions as mappers."""

        # Lambda that removes messages with "remove" in text
        def remove_mapper(msg, idx, msgs):
            return None if any("remove" in c.get("text", "") for c in msg.get("content", [])) else msg

        manager = MappingConversationManager(
            mapper=remove_mapper,
            preserve_first=1,
            preserve_last=1,
        )

        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Keep 1"}]},
            {"role": "user", "content": [{"text": "remove this"}]},
            {"role": "user", "content": [{"text": "Keep 2"}]},
        ]

        manager.reduce_context(mock_agent)
        assert len(mock_agent.messages) == 2
        assert mock_agent.messages[0]["content"][0]["text"] == "Keep 1"
        assert mock_agent.messages[1]["content"][0]["text"] == "Keep 2"

    def test_proactive_mapping_with_no_prunable_messages(self, manager, mock_agent):
        """Test that proactive mapping does not trigger with no prunable messages."""
        # Only 3 messages total, with preserve_first=1 and preserve_last=2
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial"}]},
            {"role": "user", "content": [{"text": "Recent 1"}]},
            {"role": "user", "content": [{"text": "Recent 2"}]},
        ]

        # Should not trigger mapping (no prunable messages)
        manager.apply_management(mock_agent)
        assert len(mock_agent.messages) == 3  # No change

    def test_proactive_mapping_with_prunable_messages(self, simple_manager, mock_agent):
        """Test that proactive mapping triggers when there are prunable messages."""
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial"}]},
            {"role": "user", "content": [{"text": "Old message"}]},
            {"role": "user", "content": [{"text": "Recent 1"}]},
            {"role": "user", "content": [{"text": "Recent 2"}]},
        ]

        # Should trigger mapping and remove "Old message"
        simple_manager.apply_management(mock_agent)
        assert len(mock_agent.messages) == 3
        assert simple_manager.removed_message_count == 1

    def test_state_persistence(self, manager):
        """Test getting and restoring state."""
        manager.removed_message_count = 5
        state = manager.get_state()

        assert state["removed_message_count"] == 5
        assert state["__name__"] == "MappingConversationManager"

        # Create a no-op mapper for testing
        def noop_mapper(msg, idx, msgs):
            return msg

        new_manager = MappingConversationManager(mapper=noop_mapper)
        new_manager.restore_from_session(state)
        assert new_manager.removed_message_count == 5

    def test_mapping_with_no_removals(self, simple_manager, mock_agent):
        """Test mapping even when no messages are removed."""
        # Create messages where nothing will be removed
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Keep 1"}]},
            {"role": "user", "content": [{"text": "Keep 2"}]},
            {"role": "user", "content": [{"text": "Keep 3"}]},
            {"role": "user", "content": [{"text": "Keep 4"}]},
        ]

        # Should complete successfully even if nothing is removed
        simple_manager.reduce_context(mock_agent)
        assert len(mock_agent.messages) == 4
        assert simple_manager.removed_message_count == 0


class TestLargeToolResultMapper:
    """Test the LargeToolResultMapper implementation."""

    @pytest.fixture
    def mapper(self):
        return LargeToolResultMapper(max_tokens=100, truncate_at=50)

    @pytest.fixture
    def mock_agent(self):
        return Mock()

    def test_initialization(self):
        """Test mapper initialization."""
        mapper = LargeToolResultMapper(max_tokens=50000, truncate_at=500, compression_template="Custom template")

        assert mapper.max_tokens == 50000
        assert mapper.truncate_at == 500
        assert mapper.compression_template == "Custom template"

    def test_does_not_map_small_tool_results(self, mapper):
        """Test that small tool results are not modified."""
        message: Message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"text": "Small result"}],
                        "status": "success",
                    }
                }
            ],
        }

        result = mapper(message, 0, [message])
        assert result == message  # Should be unchanged

    def test_compresses_large_tool_results(self, mapper):
        """Test that large tool results are compressed."""
        large_text = "A" * 1000  # Large content
        message: Message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"text": large_text}],
                        "status": "success",
                    }
                }
            ],
        }

        result = mapper(message, 0, [message])

        assert result is not None
        assert result != message  # Should be modified

        # Check compression occurred
        tool_result = result["content"][0]["toolResult"]
        assert len(tool_result["content"]) >= 1
        assert "compressed" in tool_result["content"][0]["text"].lower()

    def test_does_not_map_non_tool_messages(self, mapper):
        """Test that regular messages are not affected."""
        message: Message = {"role": "user", "content": [{"text": "Regular message"}]}

        result = mapper(message, 0, [message])
        assert result == message

    def test_estimate_tokens_for_text(self, mapper):
        """Test token estimation for text content."""
        tool_result = {
            "toolUseId": "123",
            "content": [{"text": "test " * 100}],  # 400 chars -> ~100 tokens
            "status": "success",
        }

        tokens = mapper._estimate_tool_result_tokens(tool_result)
        assert tokens > 50
        assert isinstance(tokens, int)

    def test_estimate_tokens_for_json(self, mapper):
        """Test token estimation for JSON content."""
        tool_result = {
            "toolUseId": "123",
            "content": [{"json": {"key": "value", "number": 42}}],
            "status": "success",
        }

        tokens = mapper._estimate_tool_result_tokens(tool_result)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_compress_truncates_long_text(self, mapper):
        """Test that long text is truncated."""
        tool_result = {
            "toolUseId": "123",
            "content": [{"text": "A" * 500}],
            "status": "success",
        }

        compressed = mapper._compress_tool_result(tool_result)

        assert compressed["toolUseId"] == "123"
        assert compressed["status"] == "success"

        # Should have compression note + truncated content
        assert len(compressed["content"]) >= 2
        assert "compressed" in compressed["content"][0]["text"].lower()
        assert "truncated" in compressed["content"][1]["text"].lower()

    def test_compress_summarizes_large_json_dict(self, mapper):
        """Test that large JSON dicts are summarized."""
        large_json = {f"key_{i}": f"value_{i}" for i in range(100)}
        tool_result = {
            "toolUseId": "123",
            "content": [{"json": large_json}],
            "status": "success",
        }

        compressed = mapper._compress_tool_result(tool_result)

        # Should have compression summary
        json_content = compressed["content"][1]["json"]
        assert json_content.get("_compressed") is True
        assert json_content.get("_type") == "dict"
        assert json_content.get("_original_keys") == 100

    def test_compress_summarizes_large_json_list(self, mapper):
        """Test that large JSON lists are summarized."""
        large_list = [f"item_{i}" for i in range(100)]
        tool_result = {
            "toolUseId": "123",
            "content": [{"json": large_list}],
            "status": "success",
        }

        compressed = mapper._compress_tool_result(tool_result)

        # Should have compression summary
        json_content = compressed["content"][1]["json"]
        assert json_content.get("_compressed") is True
        assert json_content.get("_type") == "list"
        assert json_content.get("_length") == 100

    def test_preserves_small_json(self, mapper):
        """Test that small JSON is not compressed."""
        small_json = {"key": "value"}
        tool_result = {
            "toolUseId": "123",
            "content": [{"json": small_json}],
            "status": "success",
        }

        compressed = mapper._compress_tool_result(tool_result)

        # Should preserve small JSON
        json_content = compressed["content"][1]["json"]
        assert json_content == small_json

    def test_multiple_content_items(self, mapper):
        """Test compression with multiple content items."""
        tool_result = {
            "toolUseId": "123",
            "content": [
                {"text": "A" * 100},
                {"json": {"key": "value"}},
                {"text": "B" * 100},
            ],
            "status": "success",
        }

        compressed = mapper._compress_tool_result(tool_result)

        # Should have compression note + all content items
        assert len(compressed["content"]) >= 4  # note + 3 items
