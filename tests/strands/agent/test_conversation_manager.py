import pytest

import strands
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    get_tool_result_ids,
    get_tool_use_ids,
    has_tool_result,
    has_tool_use,
)
from strands.types.exceptions import ContextWindowOverflowException


@pytest.mark.parametrize(("role", "exp_result"), [("user", True), ("assistant", False)])
def test_is_user_message(role, exp_result):
    from strands.agent.conversation_manager.sliding_window_conversation_manager import is_user_message

    tru_result = is_user_message({"role": role})
    assert tru_result == exp_result


@pytest.mark.parametrize(("role", "exp_result"), [("user", False), ("assistant", True)])
def test_is_assistant_message(role, exp_result):
    from strands.agent.conversation_manager.sliding_window_conversation_manager import is_assistant_message

    tru_result = is_assistant_message({"role": role})
    assert tru_result == exp_result


@pytest.fixture
def conversation_manager(request):
    params = {
        "window_size": 2,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return strands.agent.conversation_manager.SlidingWindowConversationManager(**params)


@pytest.mark.parametrize(
    ("conversation_manager", "messages", "expected_messages"),
    [
        # 0 - Message count under max window size - Latest assistant
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there"}]},
            ],
            [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there"}]},
            ],
        ),
        # 1 - Message count under max window size - Latest user
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
            ],
        ),
        # 2 - Remove dangling user message with no tool result
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"text": "Hello"}]},
            ],
            [],
        ),
        # 3 - Remove dangling assistant message with tool use
        (
            {"window_size": 3},
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [],
        ),
        # 4 - Remove dangling assistant message with tool use - User tool result remains
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
            ],
        ),
        # 5 - Remove dangling assistant message with tool use and user message without tool result
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "First"}]},
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Use a tool"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [
                {"role": "user", "content": [{"text": "First"}]},
                {"role": "assistant", "content": [{"text": "First response"}]},
            ],
        ),
        # 6 - Message count above max window size - Basic drop
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Second message"}]},
                {"role": "assistant", "content": [{"text": "Second response"}]},
            ],
            [
                {"role": "user", "content": [{"text": "Second message"}]},
                {"role": "assistant", "content": [{"text": "Second response"}]},
            ],
        ),
        # 7 - Message count above max window size - Remove dangling tool uses and tool results
        (
            {"window_size": 1},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "321", "name": "tool1", "input": {}}}]},
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "123", "content": [{"text": "Hello!"}], "status": "success"}}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "123", "content": [{"text": "Hello!"}], "status": "success"}}
                    ],
                },
            ],
        ),
        # 8 - Message count above max window size - Remove multiple tool use/tool result pairs
        (
            {"window_size": 1},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "789", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "789", "content": [], "status": "success"}}]},
            ],
        ),
        # 9 - Message count above max window size - Preserve tool use/tool result pairs
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
        ),
        # 10 - Test sliding window behavior - preserve tool use/result pairs across cut boundary
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
        ),
        # 11 - Test sliding window with multiple tool pairs that need preservation
        # Note: The manager prioritizes keeping tool pairs together, which may exceed window size
        (
            {"window_size": 4},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool2", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Final response"}]},
            ],
            # The manager keeps tool pairs together, resulting in 5 messages to preserve the first pair
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool2", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Final response"}]},
            ],
        ),
    ],
    indirect=["conversation_manager"],
)
def test_apply_management(conversation_manager, messages, expected_messages):
    conversation_manager.apply_management(messages)

    assert messages == expected_messages


def test_null_conversation_manager_reduce_context_raises_context_window_overflow_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = strands.agent.conversation_manager.NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()

    manager.apply_management(messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.reduce_context(messages)

    assert messages == original_messages


def test_null_conversation_manager_reduce_context_with_exception_raises_same_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = strands.agent.conversation_manager.NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()

    manager.apply_management(messages)

    with pytest.raises(RuntimeError):
        manager.reduce_context(messages, RuntimeError("test"))

    assert messages == original_messages


def test_has_tool_use():
    """Test has_tool_use helper function."""
    from strands.agent.conversation_manager.sliding_window_conversation_manager import has_tool_use

    message_with_tool_use = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "123", "name": "test", "input": {}}}],
    }
    message_without_tool_use = {"role": "assistant", "content": [{"text": "Hello"}]}
    assert has_tool_use(message_with_tool_use) is True
    assert has_tool_use(message_without_tool_use) is False


def test_has_tool_result():
    """Test has_tool_result helper function."""
    from strands.agent.conversation_manager.sliding_window_conversation_manager import has_tool_result

    message_with_tool_result = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}],
    }
    message_without_tool_result = {"role": "user", "content": [{"text": "Hello"}]}
    assert has_tool_result(message_with_tool_result) is True
    assert has_tool_result(message_without_tool_result) is False


def test_get_tool_use_ids():
    """Test get_tool_use_ids helper function."""
    from strands.agent.conversation_manager.sliding_window_conversation_manager import get_tool_use_ids

    message_with_multiple_tool_uses = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "123", "name": "test1", "input": {}}},
            {"toolUse": {"toolUseId": "456", "name": "test2", "input": {}}},
            {"text": "Some text"},
        ],
    }
    message_without_tool_use = {"role": "assistant", "content": [{"text": "Hello"}]}
    assert get_tool_use_ids(message_with_multiple_tool_uses) == ["123", "456"]
    assert get_tool_use_ids(message_without_tool_use) == []


def test_get_tool_result_ids():
    """Test get_tool_result_ids helper function."""
    from strands.agent.conversation_manager.sliding_window_conversation_manager import get_tool_result_ids

    message_with_multiple_tool_results = {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "123", "content": [], "status": "success"}},
            {"toolResult": {"toolUseId": "456", "content": [], "status": "success"}},
            {"text": "Some text"},
        ],
    }
    message_without_tool_result = {"role": "user", "content": [{"text": "Hello"}]}
    assert get_tool_result_ids(message_with_multiple_tool_results) == ["123", "456"]
    assert get_tool_result_ids(message_without_tool_result) == []


def test_reduce_context_edge_cases():
    """Test edge cases in reduce_context method."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # Test case 1: Messages with multiple content items including toolResult
    # Preserves tool results as-is
    messages = [
        {"role": "user", "content": [{"text": "First"}]},
        {
            "role": "user",
            "content": [
                {"text": "Multiple items"},
                {"toolResult": {"toolUseId": "123", "content": [], "status": "success"}},
            ],
        },
        {"role": "assistant", "content": [{"text": "Response"}]},
    ]

    manager.reduce_context(messages)

    # Should keep the last 2 messages
    assert len(messages) == 2
    assert messages[0]["content"][0]["text"] == "Multiple items"
    assert "toolResult" in messages[0]["content"][1]

    # Test case 2: Extreme case - trim_index >= len(messages)
    # Create a scenario where window_size is 2 but we need to trim from messages with length 1
    messages = [
        {"role": "user", "content": [{"text": "Only message"}]},
    ]

    # Force a reduce context when trim_index would be >= len(messages)
    # Since len(messages) = 1 and window_size = 2, trim_index = 2
    with pytest.raises(ContextWindowOverflowException, match="Unable to trim conversation context!"):
        manager.reduce_context(messages)


def test_find_safe_trim_index_orphaned_results():
    """Test _find_safe_trim_index with orphaned tool results."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=3)

    # Create messages with orphaned tool results (no matching toolUse)
    messages = [
        {"role": "user", "content": [{"text": "First"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
        # Orphaned tool result - no matching toolUse
        {"role": "user", "content": [{"toolResult": {"toolUseId": "orphan1", "content": [], "status": "success"}}]},
        {"role": "assistant", "content": [{"text": "Another response"}]},
        # Another orphaned tool result
        {"role": "user", "content": [{"toolResult": {"toolUseId": "orphan2", "content": [], "status": "success"}}]},
        {"role": "assistant", "content": [{"text": "Final response"}]},
    ]

    # Basic trim index would be 3 (6 messages - 3 window size)
    initial_trim_index = 3
    safe_index = manager._find_safe_trim_index(messages, initial_trim_index)

    # The orphaned tool result at index 2 should be OK to cut at
    # But the method should prefer index 3 (assistant message without tool use)
    assert safe_index == 3


def test_find_safe_trim_index_tool_result_before_use():
    """Test warning case where tool result appears before tool use."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=3)

    # Create an invalid scenario (shouldn't happen in practice)
    messages = [
        {"role": "user", "content": [{"text": "Start"}]},
        # Tool result before tool use (invalid)
        {"role": "user", "content": [{"toolResult": {"toolUseId": "backwards", "content": [], "status": "success"}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "backwards", "name": "tool", "input": {}}}]},
        {"role": "user", "content": [{"text": "End"}]},
    ]

    # This should trigger the warning in _find_safe_trim_index
    initial_trim_index = 1
    safe_index = manager._find_safe_trim_index(messages, initial_trim_index)
    # Should still return a valid index
    assert safe_index >= 0


def test_find_safe_trim_index_extreme_no_good_cut():
    """Test _find_safe_trim_index when initial trim is beyond messages."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # Create a scenario where initial trim index is already beyond messages length
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "content": [], "status": "success"}}]},
    ]

    # Initial trim index is 10 (way beyond the 2 messages)
    initial_trim_index = 10
    safe_index = manager._find_safe_trim_index(messages, initial_trim_index)

    # Should still be beyond messages length since there's no good cut
    assert safe_index >= len(messages)


def test_reduce_context_with_orphaned_tool_result_at_start():
    """Test reduce_context when the safe trim point starts with an orphaned tool result."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # Create messages where the trim will start with an orphaned tool result
    messages = [
        {"role": "user", "content": [{"text": "First"}]},
        {"role": "assistant", "content": [{"text": "Response"}]},
        # This will be the first message after trim - it's an orphaned tool result
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "orphan", "content": [{"text": "Result text"}], "status": "success"}}
            ],
        },
        {"role": "assistant", "content": [{"text": "Final"}]},
    ]

    # Reduce context - preserves the toolResult
    manager.reduce_context(messages)

    # Check that the messages are trimmed correctly
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    # The toolResult should be preserved
    assert "toolResult" in messages[0]["content"][0]
    assert messages[0]["content"][0]["toolResult"]["toolUseId"] == "orphan"


def test_reduce_context_safe_trim_beyond_messages():
    """Test reduce_context when it preserves tool use/result pairs."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # Create a scenario where there's a tool use/result pair plus an orphaned result
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "content": [], "status": "success"}}]},
        # This is an orphaned tool result (no matching toolUse)
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "orphan", "content": [{"text": "orphaned"}], "status": "success"}}
            ],
        },
    ]

    # Reduce context - should preserve tool use/result pairs
    manager.reduce_context(messages)

    # Should keep the tool use/result pair to maintain integrity
    assert len(messages) >= 2
    # The pair should be preserved together
    if len(messages) >= 2:
        assert has_tool_use(messages[0])
        assert has_tool_result(messages[1])
        assert get_tool_use_ids(messages[0])[0] == get_tool_result_ids(messages[1])[0]


def test_find_safe_trim_index_fallback_to_basic_trim():
    """Test that we correctly handle the case where any trim breaks a tool pair."""
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # All messages are tool pairs - any trim breaks a pair
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool1", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "1", "content": [], "status": "success"}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "2", "name": "tool2", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "2", "content": [], "status": "success"}}]},
    ]

    # The basic trim index would be 2 (window_size=2, len=4)
    # This cuts between the two tool pairs, which is a valid cut point
    manager.reduce_context(messages)

    # Window size is respected - only 2 messages remain
    assert len(messages) == 2

    # First tool pair was removed, second pair preserved
    assert messages[0]["content"][0]["toolUse"]["toolUseId"] == "2"
    assert messages[1]["content"][0]["toolResult"]["toolUseId"] == "2"


def test_find_safe_trim_index_warning_scenario():
    """Test the warning scenario where safe_index >= len(messages)."""
    from unittest.mock import patch

    # Create a custom manager with a specific _find_safe_trim_index implementation
    # that returns an index >= len(messages) to trigger the warning
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=2)

    # Create a message list longer than window size
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi"}]},
        {"role": "user", "content": [{"text": "How are you?"}]},
        {"role": "assistant", "content": [{"text": "I'm good"}]},
    ]

    # Mock _find_safe_trim_index to return a value >= len(messages)
    with patch.object(manager, "_find_safe_trim_index", return_value=5):
        with patch("strands.agent.conversation_manager.sliding_window_conversation_manager.logger") as mock_logger:
            # This should trigger the warning and use basic trim
            manager.reduce_context(messages)

            # Warning should be logged
            mock_logger.warning.assert_called_with(
                "safe_trim_index=<%d>, messages_length=<%d> | could not find safe trim point | "
                "falling back to basic trim index",
                5,
                4,
            )

            # Should use basic trim index (2) instead of the invalid index (5)
            # So 2 messages remain (last 2 based on window size)
            assert len(messages) == 2
            assert messages[0]["content"][0]["text"] == "How are you?"
            assert messages[1]["content"][0]["text"] == "I'm good"


def test_find_safe_trim_index_tool_result_before_use_with_warning():
    """Test that warning is logged when tool result appears before tool use."""
    from unittest.mock import patch

    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(window_size=3)

    # Create an invalid scenario where tool result comes before tool use
    # and the safe_index will fall between them
    messages = [
        {"role": "user", "content": [{"text": "Start"}]},
        # Tool result at index 1 (before its corresponding tool use)
        {"role": "user", "content": [{"toolResult": {"toolUseId": "backwards", "content": [], "status": "success"}}]},
        {"role": "assistant", "content": [{"text": "Middle message"}]},  # Index 2 - this will be our safe_index
        # Tool use at index 3 (after its corresponding tool result)
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "backwards", "name": "tool", "input": {}}}]},
        {"role": "user", "content": [{"text": "End"}]},
    ]

    # Initial trim index would be 2 (5 messages - 3 window size)
    # This will make safe_index = 2, which is between result_idx (1) and use_idx (3)
    initial_trim_index = 2

    with patch("strands.agent.conversation_manager.sliding_window_conversation_manager.logger") as mock_logger:
        safe_index = manager._find_safe_trim_index(messages, initial_trim_index)

        # Verify the warning was logged with the correct message
        mock_logger.warning.assert_called_with("tool_id=<%s> | found toolResult before toolUse", "backwards")

        # The method should still return a valid index
        assert safe_index == 2  # Should use the initial trim index
