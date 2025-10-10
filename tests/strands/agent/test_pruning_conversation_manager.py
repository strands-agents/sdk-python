"""Tests for PruningConversationManager class."""

from typing import Optional
from unittest.mock import Mock, patch

import pytest

from strands.agent.conversation_manager.pruning_conversation_manager import (
    MessageContext,
    PruningConversationManager,
    PruningStrategy,
)
from strands.agent.conversation_manager.strategies.tool_result_pruning import LargeToolResultPruningStrategy
from strands.types.content import Message
from strands.types.exceptions import ContextWindowOverflowException


class TestPruningStrategy(PruningStrategy):
    """Simple test strategy that prunes messages based on content."""

    def should_prune_message(self, message: Message, context: MessageContext) -> bool:
        """Prune messages that contain 'Old message' in their text."""
        for content in message.get("content", []):
            if "text" in content and "Old message" in content["text"]:
                return True
        return False

    def prune_message(self, message: Message, agent) -> Optional[Message]:
        """Remove the message entirely."""
        return None

    def get_strategy_name(self) -> str:
        """Get the name of this pruning strategy."""
        return "TestPruningStrategy"


class TestPruningConversationManager:
    """Test the PruningConversationManager class."""

    @pytest.fixture
    def strategies(self):
        return [
            LargeToolResultPruningStrategy(max_tool_result_tokens=100),
        ]

    @pytest.fixture
    def test_strategies(self):
        """Strategies that can prune regular text messages for testing."""
        return [
            TestPruningStrategy(),
        ]

    @pytest.fixture
    def manager(self, strategies):
        return PruningConversationManager(
            pruning_strategies=strategies, preserve_recent_messages=2, context_window_size=10000
        )

    @pytest.fixture
    def test_manager(self, test_strategies):
        """Manager with test strategies for regular message pruning."""
        return PruningConversationManager(
            pruning_strategies=test_strategies, preserve_recent_messages=2, context_window_size=10000
        )

    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.messages = []
        return agent

    def test_initialization(self, strategies):
        """Test manager initialization with parameters."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            preserve_recent_messages=5,
            preserve_initial_messages=2,
            enable_proactive_pruning=False,
            pruning_threshold=0.8,
            context_window_size=150000,
        )

        assert manager.pruning_strategies == strategies
        assert manager.preserve_recent_messages == 5
        assert manager.preserve_initial_messages == 2
        assert manager.enable_proactive_pruning is False
        assert manager.pruning_threshold == 0.8
        assert manager.context_window_size == 150000
        assert manager.removed_message_count == 0

    def test_parameter_clamping(self, strategies):
        """Test that parameters are clamped to valid ranges."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            pruning_threshold=2.0,  # Should be clamped to 1.0
        )

        assert manager.pruning_threshold == 1.0

        # Test lower bounds
        manager2 = PruningConversationManager(
            pruning_strategies=strategies,
            pruning_threshold=0.05,  # Should be clamped to 0.1
        )

        assert manager2.pruning_threshold == 0.1

    def test_reduce_context_with_empty_messages(self, manager, mock_agent):
        """Test that reduce_context raises exception with empty messages."""
        mock_agent.messages = []

        with pytest.raises(ContextWindowOverflowException, match="No messages to prune"):
            manager.reduce_context(mock_agent)

    def test_reduce_context_with_insufficient_messages(self, manager, mock_agent):
        """Test behavior with too few messages to prune safely."""
        mock_agent.messages = [{"role": "user", "content": [{"text": "Message 1"}]}]

        with pytest.raises(ContextWindowOverflowException, match="Insufficient messages for pruning"):
            manager.reduce_context(mock_agent)

    def test_successful_pruning(self, test_manager, mock_agent):
        """Test successful message pruning."""
        # Create messages with some that should be pruned
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Old message 1"}]},  # Age 5 - should be pruned
            {"role": "user", "content": [{"text": "Old message 2"}]},  # Age 4 - should be pruned
            {"role": "user", "content": [{"text": "Old message 3"}]},  # Age 3 - should be pruned
            {"role": "user", "content": [{"text": "Recent message 1"}]},  # Age 1 - preserved
            {"role": "user", "content": [{"text": "Recent message 2"}]},  # Age 0 - preserved
        ]

        original_count = len(mock_agent.messages)

        with patch.object(test_manager, "_validate_pruning_effectiveness", return_value=True):
            test_manager.reduce_context(mock_agent)

        # Should have fewer messages after pruning
        assert len(mock_agent.messages) < original_count
        assert test_manager.removed_message_count > 0

    def test_proactive_pruning_disabled(self, manager, mock_agent):
        """Test that proactive pruning can be disabled."""
        manager.enable_proactive_pruning = False
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(100)]

        # Should not trigger pruning
        manager.apply_management(mock_agent)

        assert len(mock_agent.messages) == 100  # No change

    def test_proactive_pruning_enabled(self, manager, mock_agent):
        """Test that proactive pruning works when enabled."""
        manager.enable_proactive_pruning = True
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(100)]

        with patch.object(manager, "_should_prune_proactively", return_value=True):
            with patch.object(manager, "reduce_context") as mock_reduce:
                manager.apply_management(mock_agent)
                mock_reduce.assert_called_once()

    def test_preserve_recent_messages(self, manager, mock_agent):
        """Test that recent messages are always preserved."""
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(10)]

        with patch.object(manager, "_validate_pruning_effectiveness", return_value=True):
            manager.reduce_context(mock_agent)

        # Should have at least preserve_recent_messages (2) remaining
        assert len(mock_agent.messages) >= manager.preserve_recent_messages

    def test_token_threshold_pruning(self, test_manager, mock_agent):
        """Test that pruning is triggered when token threshold is exceeded."""
        # Create messages that exceed the token threshold
        # With context_window_size=10000 and threshold=0.7, threshold is 7000 tokens
        large_text = "Old message with many words to exceed the token threshold. " * 50
        mock_agent.messages = (
            [{"role": "user", "content": [{"text": "Initial message"}]}]  # Preserved
            + [{"role": "user", "content": [{"text": large_text}]} for i in range(10)]  # Should be pruned
            + [
                {"role": "user", "content": [{"text": "Recent message 1"}]},  # Preserved
                {"role": "user", "content": [{"text": "Recent message 2"}]},  # Preserved
            ]
        )

        original_count = len(mock_agent.messages)

        with patch.object(test_manager, "_validate_pruning_effectiveness", return_value=True):
            test_manager.reduce_context(mock_agent)

        # Should have pruned the middle messages when threshold exceeded
        assert len(mock_agent.messages) < original_count
        # Should preserve initial and recent messages
        assert (
            len(mock_agent.messages) >= test_manager.preserve_recent_messages + test_manager.preserve_initial_messages
        )

    def test_validation_failure_handling(self, manager, mock_agent):
        """Test handling when pruning validation fails."""
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(10)]

        with patch.object(manager, "_validate_pruning_effectiveness", return_value=False):
            with patch.object(manager, "_handle_pruning_failure") as mock_handle:
                manager.reduce_context(mock_agent)
                mock_handle.assert_called_once()

    def test_pruning_effectiveness_validation(self, manager, mock_agent):
        """Test pruning effectiveness validation."""
        original_messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(10)]

        # Test successful reduction
        pruned_messages = original_messages[:5]  # 50% reduction
        assert manager._validate_pruning_effectiveness(original_messages, pruned_messages, mock_agent) is True

        # Test small reduction - still considered effective as long as there's any reduction
        pruned_messages = original_messages[:9]  # 10% reduction
        assert manager._validate_pruning_effectiveness(original_messages, pruned_messages, mock_agent) is True

        # Test no reduction
        pruned_messages = original_messages.copy()
        assert manager._validate_pruning_effectiveness(original_messages, pruned_messages, mock_agent) is False

    def test_state_management(self, manager):
        """Test state serialization and restoration."""
        # Modify some stats
        manager.removed_message_count = 15

        # Get state
        state = manager.get_state()

        assert state["removed_message_count"] == 15

        # Create new manager and restore state
        new_manager = PruningConversationManager(pruning_strategies=[])
        new_manager.restore_from_session(state)

        assert new_manager.removed_message_count == 15

    def test_should_prune_proactively(self, manager, mock_agent):
        """Test proactive pruning threshold logic."""
        # Test with few messages - should not prune (below threshold)
        # With context_window_size=10000 and threshold=0.7, we need > 7000 tokens to trigger pruning
        mock_agent.messages = [{"role": "user", "content": [{"text": "Short message"}]} for i in range(10)]
        assert manager._should_prune_proactively(mock_agent) is False

        # Test with many large messages - should prune (above threshold)
        # Create messages that will exceed the token threshold
        large_text = "This is a very long message that contains many words to increase token count. " * 100
        mock_agent.messages = [{"role": "user", "content": [{"text": large_text}]} for i in range(10)]
        assert manager._should_prune_proactively(mock_agent) is True

        # Test with empty messages
        mock_agent.messages = []
        assert manager._should_prune_proactively(mock_agent) is False

    def test_handle_pruning_failure_with_exception(self, manager, mock_agent):
        """Test pruning failure handling with original exception."""
        original_exception = ContextWindowOverflowException("Original error")

        with pytest.raises(ContextWindowOverflowException, match="Original error"):
            manager._handle_pruning_failure(mock_agent, original_exception)

    def test_handle_pruning_failure_without_exception(self, manager, mock_agent):
        """Test pruning failure handling without original exception."""
        with pytest.raises(ContextWindowOverflowException, match="Pruning failed to reduce context"):
            manager._handle_pruning_failure(mock_agent, None)

    def test_pruning_with_tool_result_strategy(self, mock_agent):
        """Test pruning with tool result compression strategy."""
        # Create a manager with tool result compression strategy
        strategies = [
            LargeToolResultPruningStrategy(max_tool_result_tokens=50),  # Compresses
        ]
        manager = PruningConversationManager(
            pruning_strategies=strategies, preserve_recent_messages=2, context_window_size=10000
        )

        # Create messages with large tool results
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Regular message"}]},
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "123",
                            "content": [{"text": "A" * 1000}],  # Large result - should be compressed
                            "status": "success",
                        }
                    }
                ],
            },
            {"role": "user", "content": [{"text": "Recent message 1"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Recent message 2"}]},  # Should be preserved
        ]

        with patch.object(manager, "_validate_pruning_effectiveness", return_value=True):
            manager.reduce_context(mock_agent)

        # Recent messages should be preserved
        assert len(mock_agent.messages) >= manager.preserve_recent_messages

    def test_error_propagation_during_pruning(self, manager, mock_agent):
        """Test that errors during pruning are properly propagated."""
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(10)]

        # Mock an error during message pruning
        with patch.object(manager, "_prune_messages", side_effect=RuntimeError("Pruning error")):
            original_exception = ContextWindowOverflowException("Original")

            with pytest.raises(RuntimeError, match="Pruning error"):
                manager.reduce_context(mock_agent, original_exception)

    def test_preserve_initial_messages(self, strategies, mock_agent):
        """Test that initial messages are preserved when preserve_initial_messages is set."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            preserve_recent_messages=2,
            preserve_initial_messages=2,
            context_window_size=10000,
        )

        # Create 10 messages: first 2 should be preserved, last 2 should be preserved
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial message 1"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Initial message 2"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Middle message 1"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Middle message 2"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Middle message 3"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Middle message 4"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Middle message 5"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Middle message 6"}]},  # Can be pruned
            {"role": "user", "content": [{"text": "Recent message 1"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Recent message 2"}]},  # Should be preserved
        ]

        with patch.object(manager, "_validate_pruning_effectiveness", return_value=True):
            manager.reduce_context(mock_agent)

        # Should have at least 4 messages (2 initial + 2 recent)
        assert len(mock_agent.messages) >= 4

        # Check that initial messages are preserved
        assert mock_agent.messages[0]["content"][0]["text"] == "Initial message 1"
        assert mock_agent.messages[1]["content"][0]["text"] == "Initial message 2"

        # Check that recent messages are preserved
        assert mock_agent.messages[-2]["content"][0]["text"] == "Recent message 1"
        assert mock_agent.messages[-1]["content"][0]["text"] == "Recent message 2"

    def test_preserve_initial_and_recent_with_insufficient_messages(self, strategies, mock_agent):
        """Test behavior when total preserved messages exceed available messages."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            preserve_recent_messages=3,
            preserve_initial_messages=3,
            context_window_size=10000,
        )

        # Only 5 messages, but we want to preserve 6 (3 initial + 3 recent)
        mock_agent.messages = [{"role": "user", "content": [{"text": f"Message {i}"}]} for i in range(5)]

        with pytest.raises(ContextWindowOverflowException, match="Insufficient messages for pruning"):
            manager.reduce_context(mock_agent)

    def test_preserve_initial_messages_default_one(self, strategies):
        """Test that preserve_initial_messages defaults to 1."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            preserve_recent_messages=2,
        )

        assert manager.preserve_initial_messages == 1

    def test_context_window_size_configuration(self, strategies):
        """Test that context window size can be configured."""
        manager = PruningConversationManager(
            pruning_strategies=strategies,
            context_window_size=50000,
            pruning_threshold=0.8,
        )

        assert manager.context_window_size == 50000
        assert manager.pruning_threshold == 0.8

    def test_pruning_context_token_calculation(self, manager, mock_agent):
        """Test that PruningContext is used for token calculation."""
        messages = [
            {"role": "user", "content": [{"text": "Hello world"}]},
            {"role": "assistant", "content": [{"text": "Hi there how are you"}]},
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "123",
                            "content": [{"text": "Tool result text"}],
                            "status": "success",
                        }
                    }
                ],
            },
        ]
        mock_agent.messages = messages

        # Test that _should_prune_proactively uses PruningContext
        result = manager._should_prune_proactively(mock_agent)

        # Should not prune with small messages and default threshold
        assert result is False

    def test_aggressive_pruning_when_threshold_exceeded(self, test_manager, mock_agent):
        """Test that all eligible messages are pruned when token threshold is exceeded."""
        # Create messages that will exceed the threshold
        large_text = "Old message with many words to exceed the token threshold. " * 50
        mock_agent.messages = [
            {"role": "user", "content": [{"text": "Initial message"}]},  # Should be preserved
            {"role": "user", "content": [{"text": large_text}]},  # Should be pruned
            {"role": "user", "content": [{"text": large_text}]},  # Should be pruned
            {"role": "user", "content": [{"text": large_text}]},  # Should be pruned
            {"role": "user", "content": [{"text": "Recent message 1"}]},  # Should be preserved
            {"role": "user", "content": [{"text": "Recent message 2"}]},  # Should be preserved
        ]

        original_count = len(mock_agent.messages)

        with patch.object(test_manager, "_validate_pruning_effectiveness", return_value=True):
            test_manager.reduce_context(mock_agent)

        # Should have aggressively pruned middle messages
        assert len(mock_agent.messages) < original_count
        # Should preserve initial and recent messages
        expected_preserved = test_manager.preserve_initial_messages + test_manager.preserve_recent_messages
        assert len(mock_agent.messages) >= expected_preserved

    def test_token_based_proactive_pruning_integration(self, manager, mock_agent):
        """Test the complete token-based proactive pruning workflow."""
        # Create messages that exceed the threshold (0.7 * 10000 = 7000 tokens)
        large_text = "This is a message with many words to simulate a large conversation history. " * 50
        mock_agent.messages = [{"role": "user", "content": [{"text": large_text}]} for i in range(10)]

        # Should trigger proactive pruning
        assert manager._should_prune_proactively(mock_agent) is True

        # Test that apply_management calls reduce_context when proactive pruning is enabled
        manager.enable_proactive_pruning = True
        with patch.object(manager, "reduce_context") as mock_reduce:
            manager.apply_management(mock_agent)
            mock_reduce.assert_called_once()
