"""Tests for pruning strategies."""

from unittest.mock import Mock

import pytest

from strands.agent.conversation_manager import PruningStrategy
from strands.agent.conversation_manager.strategies.tool_result_pruning import LargeToolResultPruningStrategy
from strands.types.content import Message


class TestPruningStrategy:
    """Test the abstract PruningStrategy interface."""

    def test_abstract_methods_must_be_implemented(self):
        """Test that PruningStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PruningStrategy()


class TestLargeToolResultPruningStrategy:
    """Test the LargeToolResultPruningStrategy implementation."""

    @pytest.fixture
    def strategy(self):
        return LargeToolResultPruningStrategy(max_tool_result_tokens=100)

    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.messages = []
        return agent

    def test_should_prune_message_with_large_tool_result(self, strategy):
        """Test that large tool results are identified for pruning."""
        message: Message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"text": "A" * 1000}],  # Large content
                        "status": "success",
                    }
                }
            ],
        }
        context = {"has_tool_result": True}

        assert strategy.should_prune_message(message, context) is True

    def test_should_not_prune_message_with_small_tool_result(self, strategy):
        """Test that small tool results are not pruned."""
        message: Message = {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Small result"}], "status": "success"}}
            ],
        }
        context = {"has_tool_result": True}

        assert strategy.should_prune_message(message, context) is False

    def test_should_not_prune_message_without_tool_result(self, strategy):
        """Test that messages without tool results are not pruned."""
        message: Message = {"role": "user", "content": [{"text": "Regular message"}]}
        context = {"has_tool_result": False}

        assert strategy.should_prune_message(message, context) is False

    def test_prune_message_compresses_large_tool_result(self, strategy, mock_agent):
        """Test that large tool results are properly compressed."""
        message: Message = {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "123", "content": [{"text": "A" * 1000}], "status": "success"}}],
        }

        pruned = strategy.prune_message(message, mock_agent)

        assert pruned is not None
        assert pruned["role"] == "user"

        # Check that compression occurred
        tool_result = pruned["content"][0]["toolResult"]
        assert len(tool_result["content"]) >= 1
        assert "compressed" in tool_result["content"][0]["text"].lower()

    def test_estimate_tool_result_tokens(self, strategy):
        """Test token estimation for tool results."""
        tool_result = {
            "toolUseId": "123",
            "content": [
                {"text": "This is a test message with ten words total"},
                {"json": {"key": "value", "number": 42}},
            ],
            "status": "success",
        }

        tokens = strategy._estimate_tool_result_tokens(tool_result)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_simple_compress_tool_result(self, strategy):
        """Test simple compression of tool results."""
        tool_result = {
            "toolUseId": "123",
            "content": [{"text": "A" * 500}],  # Long text
            "status": "success",
        }

        compressed = strategy._simple_compress_tool_result(tool_result)

        assert compressed["toolUseId"] == "123"
        assert compressed["status"] == "success"
        assert len(compressed["content"]) >= 1
        assert "compressed" in compressed["content"][0]["text"].lower()

    def test_get_strategy_name(self, strategy):
        """Test strategy name retrieval."""
        assert strategy.get_strategy_name() == "LargeToolResultPruningStrategy"
