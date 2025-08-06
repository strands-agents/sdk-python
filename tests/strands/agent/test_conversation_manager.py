import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.types.content import Message
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)


@pytest.fixture
def conversation_manager(request):
    params = {
        "window_size": 2,
        "should_truncate_results": False,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return SlidingWindowConversationManager(**params)


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
        # 2 - Keep user message
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"text": "Hello"}]},
            ],
            [{"role": "user", "content": [{"text": "Hello"}]}],
        ),
        # 3 - Keep dangling assistant message with tool use
        (
            {"window_size": 3},
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [{"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]}],
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
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
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
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Use a tool"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
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
        # 7 - Message count above max window size - Preserve tool use/tool result pairs
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
        # 8 - Test sliding window behavior - preserve tool use/result pairs across cut boundary
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
        # 9 - Test sliding window with multiple tool pairs that need preservation
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
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool2", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Final response"}]},
            ],
        ),
    ],
    indirect=["conversation_manager"],
)
def test_apply_management(conversation_manager, messages, expected_messages):
    test_agent = Agent(messages=messages)
    conversation_manager.apply_management(test_agent)

    assert messages == expected_messages


def test_sliding_window_conversation_manager_with_untrimmable_history_raises_context_window_overflow_exception():
    manager = SlidingWindowConversationManager(1, False)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "789", "content": [], "status": "success"}}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.apply_management(test_agent)

    assert messages == original_messages


def test_sliding_window_conversation_manager_with_tool_results_truncated():
    manager = SlidingWindowConversationManager(1)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "789", "content": [{"text": "large input"}], "status": "success"}}
            ],
        },
    ]
    test_agent = Agent(messages=messages)

    manager.reduce_context(test_agent)

    expected_messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "789",
                        "content": [{"text": "The tool result was too large!"}],
                        "status": "error",
                    }
                }
            ],
        },
    ]

    assert messages == expected_messages


@pytest.mark.asyncio
async def test_sliding_window_conversation_manager_handle_token_limit_reached():
    """Test that SlidingWindowConversationManager handles token limit recovery."""
    manager = SlidingWindowConversationManager()
    test_agent = Agent()
    initial_message_count = len(test_agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"text": "I'll help you with that."},
            {"toolUse": {"name": "calculator", "input": {}, "toolUseId": ""}},  # Missing toolUseId
        ],
    }

    test_exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await manager.handle_token_limit_reached(test_agent, test_exception)

    # Should add one corrected message
    assert len(test_agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = test_agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 2

    # First content block should be preserved
    assert corrected_message["content"][0] == {"text": "I'll help you with that."}

    # Second content block should be replaced with error message
    assert "text" in corrected_message["content"][1]
    assert "calculator" in corrected_message["content"][1]["text"]
    assert "incomplete due to maximum token limits" in corrected_message["content"][1]["text"]


def test_null_conversation_manager_reduce_context_raises_context_window_overflow_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    manager.apply_management(test_agent)

    with pytest.raises(ContextWindowOverflowException):
        manager.reduce_context(messages)

    assert messages == original_messages


def test_null_conversation_manager_reduce_context_with_exception_raises_same_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    manager.apply_management(test_agent)

    with pytest.raises(RuntimeError):
        manager.reduce_context(messages, RuntimeError("test"))

    assert messages == original_messages


def test_null_conversation_does_not_restore_with_incorrect_state():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()

    with pytest.raises(ValueError):
        manager.restore_from_session({})


@pytest.mark.asyncio
async def test_summarizing_conversation_manager_handle_token_limit_reached():
    """Test that SummarizingConversationManager handles token limit recovery."""
    from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager

    manager = SummarizingConversationManager()
    test_agent = Agent()
    initial_message_count = len(test_agent.messages)

    incomplete_message: Message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"name": "", "input": {}, "toolUseId": "123"}},  # Missing name
        ],
    }

    test_exception = MaxTokensReachedException(message="Token limit reached", incomplete_message=incomplete_message)

    await manager.handle_token_limit_reached(test_agent, test_exception)

    # Should add one corrected message
    assert len(test_agent.messages) == initial_message_count + 1

    # Check the corrected message content
    corrected_message = test_agent.messages[-1]
    assert corrected_message["role"] == "assistant"
    assert len(corrected_message["content"]) == 1

    # Content should be replaced with error message using <unknown>
    assert "text" in corrected_message["content"][0]
    assert "<unknown>" in corrected_message["content"][0]["text"]
    assert "incomplete due to maximum token limits" in corrected_message["content"][0]["text"]


@pytest.mark.asyncio
async def test_null_conversation_manager_handle_token_limit_reached_raises_exception():
    """Test that NullConversationManager raises the provided exception."""
    manager = NullConversationManager()
    test_agent = Agent()
    test_message: Message = {
        "role": "assistant",
        "content": [{"text": "Hello"}],
    }
    test_exception = MaxTokensReachedException(message="test", incomplete_message=test_message)

    with pytest.raises(MaxTokensReachedException):
        await manager.handle_token_limit_reached(test_agent, test_exception)
