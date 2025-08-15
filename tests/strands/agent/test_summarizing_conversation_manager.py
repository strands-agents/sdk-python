from typing import cast
from unittest.mock import Mock, patch

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager
from strands.types.content import Messages
from strands.types.exceptions import ContextWindowOverflowException
from tests.fixtures.mocked_model_provider import MockedModelProvider


class MockAgent:
    """Mock agent for testing summarization."""

    def __init__(self, summary_response="This is a summary of the conversation."):
        self.summary_response = summary_response
        self.system_prompt = None
        self.messages = []
        self.model = Mock()
        self.call_tracker = Mock()

    def __call__(self, prompt):
        """Mock agent call that returns a summary."""
        self.call_tracker(prompt)
        result = Mock()
        result.message = {"role": "assistant", "content": [{"text": self.summary_response}]}
        return result


def create_mock_agent(summary_response="This is a summary of the conversation.") -> "Agent":
    """Factory function that returns a properly typed MockAgent."""
    return cast("Agent", MockAgent(summary_response))


@pytest.fixture
def mock_agent():
    """Fixture for mock agent."""
    return create_mock_agent()


@pytest.fixture
def summarizing_manager():
    """Fixture for summarizing conversation manager with default settings."""
    return SummarizingConversationManager(
        summary_ratio=0.5,
        preserve_recent_messages=2,
    )


def test_init_default_values():
    """Test initialization with default values."""
    manager = SummarizingConversationManager()

    assert manager.summarization_agent is None
    assert manager.summary_ratio == 0.3
    assert manager.preserve_recent_messages == 10


def test_init_clamps_summary_ratio():
    """Test that summary_ratio is clamped to valid range."""
    # Test lower bound
    manager = SummarizingConversationManager(summary_ratio=0.05)
    assert manager.summary_ratio == 0.1

    # Test upper bound
    manager = SummarizingConversationManager(summary_ratio=0.95)
    assert manager.summary_ratio == 0.8


def test_reduce_context_raises_when_no_agent():
    """Test that reduce_context raises exception when agent has no messages."""
    manager = SummarizingConversationManager()

    # Create a mock agent with no messages
    mock_agent = Mock()
    empty_messages: Messages = []
    mock_agent.messages = empty_messages

    with pytest.raises(ContextWindowOverflowException, match="insufficient messages for summarization"):
        manager.reduce_context(mock_agent)


def test_reduce_context_with_summarization(summarizing_manager, mock_agent):
    """Test reduce_context with summarization enabled."""
    test_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
        {"role": "assistant", "content": [{"text": "Response 2"}]},
        {"role": "user", "content": [{"text": "Message 3"}]},
        {"role": "assistant", "content": [{"text": "Response 3"}]},
    ]
    mock_agent.messages = test_messages

    summarizing_manager.reduce_context(mock_agent)

    # Should have: 1 summary message + 2 preserved recent messages + remaining from summarization
    assert len(mock_agent.messages) == 4

    # First message should be the summary
    assert mock_agent.messages[0]["role"] == "assistant"
    first_content = mock_agent.messages[0]["content"][0]
    assert "text" in first_content and "This is a summary of the conversation." in first_content["text"]

    # Recent messages should be preserved
    assert "Message 3" in str(mock_agent.messages[-2]["content"])
    assert "Response 3" in str(mock_agent.messages[-1]["content"])


def test_reduce_context_too_few_messages_raises_exception(summarizing_manager, mock_agent):
    """Test that reduce_context raises exception when there are too few messages to summarize effectively."""
    # Create a scenario where calculation results in 0 messages to summarize
    manager = SummarizingConversationManager(
        summary_ratio=0.1,  # Very small ratio
        preserve_recent_messages=5,  # High preservation
    )

    insufficient_test_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
    ]
    mock_agent.messages = insufficient_test_messages  # 5 messages, preserve_recent_messages=5, so nothing to summarize

    with pytest.raises(ContextWindowOverflowException, match="insufficient messages for summarization"):
        manager.reduce_context(mock_agent)


def test_reduce_context_insufficient_messages_for_summarization(mock_agent):
    """Test reduce_context when there aren't enough messages to summarize."""
    manager = SummarizingConversationManager(
        summary_ratio=0.5,
        preserve_recent_messages=3,
    )

    insufficient_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
    ]
    mock_agent.messages = insufficient_messages

    # This should raise an exception since there aren't enough messages to summarize
    with pytest.raises(ContextWindowOverflowException, match="insufficient messages for summarization"):
        manager.reduce_context(mock_agent)


def test_reduce_context_raises_on_summarization_failure():
    """Test that reduce_context raises exception when summarization fails."""
    # Create an agent that will fail
    failing_agent = Mock()
    failing_agent.side_effect = Exception("Agent failed")
    failing_agent.system_prompt = None
    failing_agent_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
        {"role": "assistant", "content": [{"text": "Response 2"}]},
    ]
    failing_agent.messages = failing_agent_messages

    manager = SummarizingConversationManager(
        summary_ratio=0.5,
        preserve_recent_messages=1,
    )

    with patch("strands.agent.conversation_manager.summarizing_conversation_manager.logger") as mock_logger:
        with pytest.raises(Exception, match="Agent failed"):
            manager.reduce_context(failing_agent)

        # Should log the error
        mock_logger.error.assert_called_once()


def test_generate_summary(summarizing_manager, mock_agent):
    """Test the _generate_summary method."""
    test_messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    summary = summarizing_manager._generate_summary(test_messages, mock_agent)

    summary_content = summary["content"][0]
    assert "text" in summary_content and summary_content["text"] == "This is a summary of the conversation."


def test_generate_summary_with_tool_content(summarizing_manager, mock_agent):
    """Test summary generation with tool use and results."""
    tool_messages: Messages = [
        {"role": "user", "content": [{"text": "Use a tool"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}}
            ],
        },
    ]

    summary = summarizing_manager._generate_summary(tool_messages, mock_agent)

    summary_content = summary["content"][0]
    assert "text" in summary_content and summary_content["text"] == "This is a summary of the conversation."


def test_generate_summary_raises_on_agent_failure():
    """Test that _generate_summary raises exception when agent fails."""
    failing_agent = Mock()
    failing_agent.side_effect = Exception("Agent failed")
    failing_agent.system_prompt = None
    empty_failing_messages: Messages = []
    failing_agent.messages = empty_failing_messages

    manager = SummarizingConversationManager()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    # Should raise the exception from the agent
    with pytest.raises(Exception, match="Agent failed"):
        manager._generate_summary(messages, failing_agent)


def test_adjust_split_point_for_tool_pairs(summarizing_manager):
    """Test that the split point is adjusted to avoid breaking ToolUse/ToolResult pairs."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "123",
                        "content": [{"text": "Tool output"}],
                        "status": "success",
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "Response after tool"}]},
    ]

    # If we try to split at message 2 (the ToolResult), it should move forward to message 3
    adjusted_split = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 2)
    assert adjusted_split == 3  # Should move to after the ToolResult

    # If we try to split at message 3, it should be fine (no tool issues)
    adjusted_split = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 3)
    assert adjusted_split == 3

    # If we try to split at message 1 (toolUse with following toolResult), it should be valid
    adjusted_split = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 1)
    assert adjusted_split == 1  # Should be valid because toolResult follows


def test_apply_management_no_op(summarizing_manager, mock_agent):
    """Test apply_management does not modify messages (no-op behavior)."""
    apply_test_messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
        {"role": "user", "content": [{"text": "More messages"}]},
        {"role": "assistant", "content": [{"text": "Even more"}]},
    ]
    mock_agent.messages = apply_test_messages
    original_messages = mock_agent.messages.copy()

    summarizing_manager.apply_management(mock_agent)

    # Should never modify messages - summarization only happens on context overflow
    assert mock_agent.messages == original_messages


def test_init_with_custom_parameters():
    """Test initialization with custom parameters."""
    mock_agent = create_mock_agent()

    manager = SummarizingConversationManager(
        summary_ratio=0.4,
        preserve_recent_messages=5,
        summarization_agent=mock_agent,
    )
    assert manager.summary_ratio == 0.4
    assert manager.preserve_recent_messages == 5
    assert manager.summarization_agent == mock_agent
    assert manager.summarization_system_prompt is None


def test_init_with_both_agent_and_prompt_raises_error():
    """Test that providing both agent and system prompt raises ValueError."""
    mock_agent = create_mock_agent()
    custom_prompt = "Custom summarization prompt"

    with pytest.raises(ValueError, match="Cannot provide both summarization_agent and summarization_system_prompt"):
        SummarizingConversationManager(
            summarization_agent=mock_agent,
            summarization_system_prompt=custom_prompt,
        )


def test_uses_summarization_agent_when_provided():
    """Test that summarization_agent is used when provided."""
    summary_agent = create_mock_agent("Custom summary from dedicated agent")
    manager = SummarizingConversationManager(summarization_agent=summary_agent)

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    parent_agent = create_mock_agent("Parent agent summary")
    summary = manager._generate_summary(messages, parent_agent)

    # Should use the dedicated summarization agent, not the parent agent
    summary_content = summary["content"][0]
    assert "text" in summary_content and summary_content["text"] == "Custom summary from dedicated agent"

    # Assert that the summarization agent was called
    summary_agent.call_tracker.assert_called_once()


def test_uses_parent_agent_when_no_summarization_agent():
    """Test that parent agent is used when no summarization_agent is provided."""
    manager = SummarizingConversationManager()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    parent_agent = create_mock_agent("Parent agent summary")
    summary = manager._generate_summary(messages, parent_agent)

    # Should use the parent agent
    summary_content = summary["content"][0]
    assert "text" in summary_content and summary_content["text"] == "Parent agent summary"

    # Assert that the parent agent was called
    parent_agent.call_tracker.assert_called_once()


def test_uses_custom_system_prompt():
    """Test that custom system prompt is used when provided."""
    custom_prompt = "Custom system prompt for summarization"
    manager = SummarizingConversationManager(summarization_system_prompt=custom_prompt)
    mock_agent = create_mock_agent()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    # Capture the agent's system prompt changes
    original_prompt = mock_agent.system_prompt
    manager._generate_summary(messages, mock_agent)

    # The agent's system prompt should be restored after summarization
    assert mock_agent.system_prompt == original_prompt


def test_agent_state_restoration():
    """Test that agent state is properly restored after summarization."""
    manager = SummarizingConversationManager()
    mock_agent = create_mock_agent()

    # Set initial state
    original_system_prompt = "Original system prompt"
    original_messages: Messages = [{"role": "user", "content": [{"text": "Original message"}]}]
    mock_agent.system_prompt = original_system_prompt
    mock_agent.messages = original_messages.copy()

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    manager._generate_summary(messages, mock_agent)

    # State should be restored
    assert mock_agent.system_prompt == original_system_prompt
    assert mock_agent.messages == original_messages


def test_agent_state_restoration_on_exception():
    """Test that agent state is restored even when summarization fails."""
    manager = SummarizingConversationManager()

    # Create an agent that fails during summarization
    mock_agent = Mock()
    mock_agent.system_prompt = "Original prompt"
    agent_messages: Messages = [{"role": "user", "content": [{"text": "Original"}]}]
    mock_agent.messages = agent_messages
    mock_agent.side_effect = Exception("Summarization failed")

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]

    # Should restore state even on exception
    with pytest.raises(Exception, match="Summarization failed"):
        manager._generate_summary(messages, mock_agent)

    # State should still be restored
    assert mock_agent.system_prompt == "Original prompt"


def test_reduce_context_tool_pair_adjustment_works_with_forward_search():
    """Test that tool pair adjustment works correctly with the forward-search logic."""
    manager = SummarizingConversationManager(
        summary_ratio=0.5,
        preserve_recent_messages=1,
    )

    mock_agent = create_mock_agent()
    # Create messages where the split point would be adjusted to 0 due to tool pairs
    tool_pair_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}}
            ],
        },
        {"role": "user", "content": [{"text": "Latest message"}]},
    ]
    mock_agent.messages = tool_pair_messages

    # With 3 messages, preserve_recent_messages=1, summary_ratio=0.5:
    # messages_to_summarize_count = (3 - 1) * 0.5 = 1
    # But split point adjustment will move forward from the toolUse, potentially increasing count
    manager.reduce_context(mock_agent)
    # Should have summary + remaining messages
    assert len(mock_agent.messages) == 2

    # First message should be the summary
    assert mock_agent.messages[0]["role"] == "assistant"
    summary_content = mock_agent.messages[0]["content"][0]
    assert "text" in summary_content and "This is a summary of the conversation." in summary_content["text"]

    # Last message should be the preserved recent message
    assert mock_agent.messages[1]["role"] == "user"
    assert mock_agent.messages[1]["content"][0]["text"] == "Latest message"


def test_adjust_split_point_exceeds_message_length(summarizing_manager):
    """Test that split point exceeding message array length raises exception."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
    ]

    # Try to split at point 5 when there are only 2 messages
    with pytest.raises(ContextWindowOverflowException, match="Split point exceeds message array length"):
        summarizing_manager._adjust_split_point_for_tool_pairs(messages, 5)


def test_adjust_split_point_equals_message_length(summarizing_manager):
    """Test that split point equal to message array length returns unchanged."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
    ]

    # Split point equals message length (2) - should return unchanged
    result = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 2)
    assert result == 2


def test_adjust_split_point_no_tool_result_at_split(summarizing_manager):
    """Test split point that doesn't contain tool result, ensuring we reach return split_point."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
    ]

    # Split point message is not a tool result, so it should directly return split_point
    result = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 1)
    assert result == 1


def test_adjust_split_point_tool_result_without_tool_use(summarizing_manager):
    """Test that having tool results without tool uses raises exception."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}}
            ],
        },
    ]

    # Has tool result but no tool use - invalid state
    with pytest.raises(ContextWindowOverflowException, match="Unable to trim conversation context!"):
        summarizing_manager._adjust_split_point_for_tool_pairs(messages, 1)


def test_adjust_split_point_tool_result_moves_to_end(summarizing_manager):
    """Test tool result at split point moves forward to valid position at end."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}}
            ],
        },
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "different_tool", "input": {}}}]},
    ]

    # Split at message 2 (toolResult) - will move forward to message 3 (toolUse at end is valid)
    result = summarizing_manager._adjust_split_point_for_tool_pairs(messages, 2)
    assert result == 3


def test_adjust_split_point_tool_result_no_forward_position(summarizing_manager):
    """Test tool result at split point where forward search finds no valid position."""
    messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "test_tool", "input": {}}}]},
        {"role": "user", "content": [{"text": "Message between"}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "123", "content": [{"text": "Tool output"}], "status": "success"}}
            ],
        },
    ]

    # Split at message 3 (toolResult) - will try to move forward but no valid position exists
    with pytest.raises(ContextWindowOverflowException, match="Unable to trim conversation context!"):
        summarizing_manager._adjust_split_point_for_tool_pairs(messages, 3)


def test_reduce_context_adjustment_returns_zero():
    """Test that tool pair adjustment can return zero, triggering the check at line 122."""
    manager = SummarizingConversationManager(
        summary_ratio=0.5,
        preserve_recent_messages=1,
    )

    # Mock the adjustment method to return 0
    def mock_adjust(messages, split_point):
        return 0  # This should trigger the <= 0 check at line 122

    manager._adjust_split_point_for_tool_pairs = mock_adjust

    mock_agent = Mock()
    simple_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
    ]
    mock_agent.messages = simple_messages

    # The adjustment method will return 0, which should trigger line 122-123
    with pytest.raises(ContextWindowOverflowException, match="insufficient messages for summarization"):
        manager.reduce_context(mock_agent)


def test_summarizing_conversation_manager_properly_records_removed_message_count():
    mock_model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "Summary"}]},
            {"role": "assistant", "content": [{"text": "Summary"}]},
        ]
    )

    simple_messages: Messages = [
        {"role": "user", "content": [{"text": "Message 1"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 2"}]},
        {"role": "assistant", "content": [{"text": "Response 1"}]},
        {"role": "user", "content": [{"text": "Message 3"}]},
        {"role": "assistant", "content": [{"text": "Response 3"}]},
        {"role": "user", "content": [{"text": "Message 4"}]},
        {"role": "assistant", "content": [{"text": "Response 4"}]},
    ]
    agent = Agent(model=mock_model, messages=simple_messages)
    manager = SummarizingConversationManager(summary_ratio=0.5, preserve_recent_messages=1)

    assert manager._summary_message is None
    assert manager.removed_message_count == 0

    manager.reduce_context(agent)
    # Assert the oldest message is the sumamry message
    assert manager._summary_message["content"][0]["text"] == "Summary"
    # There are 8 messages in the agent messages array, since half will be summarized,
    # 4 will remain plus 1 summary message = 5
    assert (len(agent.messages)) == 5
    # Half of the messages were summarized and removed: 8/2 = 4
    assert manager.removed_message_count == 4

    manager.reduce_context(agent)
    assert manager._summary_message["content"][0]["text"] == "Summary"
    # After the first summary, 5 messages remain. Summarizing again will lead to:
    # 5 - (int(5/2)) (messages to be sumamrized) + 1 (new summary message) = 5 - 2 + 1 = 4
    assert (len(agent.messages)) == 4
    # Half of the messages were summarized and removed: int(5/2) = 2
    # However, one of the messages that was summarized was the previous summary message,
    # so we dont count this toward the total:
    # 4 (Previously removed messages) + 2 (removed messages) - 1 (Previous summary message) = 5
    assert manager.removed_message_count == 5
