import pytest

import strands
from strands.agent.agent import Agent
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
    manager = strands.agent.conversation_manager.SlidingWindowConversationManager(1)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "789", "content": [], "status": "success"}}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.apply_management(test_agent)

    assert messages == original_messages


def test_null_conversation_manager_reduce_context_raises_context_window_overflow_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = strands.agent.conversation_manager.NullConversationManager()
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
    manager = strands.agent.conversation_manager.NullConversationManager()
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
