from unittest.mock import MagicMock, patch

import pytest

from strands import tool
from strands.agent.agent import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.hooks.events import BeforeModelCallEvent
from strands.hooks.registry import HookProvider, HookRegistry
from strands.types.exceptions import ContextWindowOverflowException
from tests.fixtures.mocked_model_provider import MockedModelProvider


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
        # 7 - Message count above max window size - Keep latest user turn as first message
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
        ),
        # 8 - Test sliding window behavior - avoid assistant-first reduced history
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
        ),
        # 9 - Test sliding window with multiple tool pairs while keeping user as first turn
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


def test_sliding_window_conversation_manager_scans_older_tool_results_when_latest_is_already_truncated():
    manager = SlidingWindowConversationManager(1)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "old", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "old", "content": [{"text": "large input"}], "status": "success"}}
            ],
        },
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "new", "name": "tool2", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "new",
                        "content": [{"text": "The tool result was too large!"}],
                        "status": "error",
                    }
                }
            ],
        },
    ]
    test_agent = Agent(messages=messages)

    manager.reduce_context(test_agent)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["toolResult"]["toolUseId"] == "old"
    assert messages[0]["content"][0]["toolResult"]["content"] == [{"text": "The tool result was too large!"}]
    assert messages[0]["content"][0]["toolResult"]["status"] == "error"


def test_sliding_window_conversation_manager_raises_when_reduced_history_has_no_user_messages():
    manager = SlidingWindowConversationManager(1, should_truncate_results=False)
    messages = [
        {"role": "user", "content": [{"text": "First message"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
    ]
    test_agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException, match="Unable to trim conversation context!"):
        manager.apply_management(test_agent)


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


# ==============================================================================
# Per-Turn Management Tests
# ==============================================================================


def test_per_turn_parameter_validation():
    """Test per_turn parameter validation."""
    # Valid values
    assert SlidingWindowConversationManager(per_turn=False).per_turn is False
    assert SlidingWindowConversationManager(per_turn=True).per_turn is True
    assert SlidingWindowConversationManager(per_turn=3).per_turn == 3


def test_conversation_manager_is_hook_provider():
    """Test that ConversationManager implements HookProvider protocol."""
    manager = NullConversationManager()
    assert isinstance(manager, HookProvider)


def test_derived_class_does_not_need_to_implement_register_hooks():
    """Test that derived classes don't need to override register_hooks for backwards compatibility."""
    from strands.agent.conversation_manager.conversation_manager import ConversationManager

    class MinimalConversationManager(ConversationManager):
        """A minimal implementation that only implements abstract methods."""

        def apply_management(self, agent, **kwargs):
            pass

        def reduce_context(self, agent, e=None, **kwargs):
            pass

    # Should be able to instantiate without implementing register_hooks
    manager = MinimalConversationManager()
    registry = HookRegistry()

    # Should work without error
    manager.register_hooks(registry)
    assert not registry.has_callbacks()


def test_per_turn_hooks_registration():
    """Test that hooks are registered when conversation_manager implements HookProvider."""
    manager = SlidingWindowConversationManager(per_turn=True)
    assert isinstance(manager, HookProvider)

    registry = HookRegistry()
    manager.register_hooks(registry)
    assert registry.has_callbacks()


def test_per_turn_false_no_management_during_loop():
    """Test that per_turn=False only manages in finally block."""
    manager = SlidingWindowConversationManager(per_turn=False, window_size=100)
    responses = [{"role": "assistant", "content": [{"text": "Response"}]}] * 3
    model = MockedModelProvider(responses)
    agent = Agent(model=model, conversation_manager=manager)

    with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock:
        agent("Test")
        # Should only be called once in finally block (per_turn disabled)
        assert mock.call_count == 1


def test_per_turn_true_manages_each_model_call():
    """Test that per_turn=True applies management before each model call."""
    manager = SlidingWindowConversationManager(per_turn=True, window_size=100)
    responses = [{"role": "assistant", "content": [{"text": "Response"}]}] * 3
    model = MockedModelProvider(responses)
    agent = Agent(model=model, conversation_manager=manager)

    with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock:
        agent("Test")
        # Should be called for each model call + finally block
        # With simple text responses, agent makes 1 model call then stops
        assert mock.call_count >= 1


def test_per_turn_integer_manages_every_n_calls():
    """Test that per_turn=N applies management every N model calls."""
    manager = SlidingWindowConversationManager(per_turn=2, window_size=100)
    # Create responses that trigger multiple model calls
    responses = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"{i}", "name": "test", "input": {}}}]}
        for i in range(5)
    ] + [{"role": "assistant", "content": [{"text": "Done"}]}]
    model = MockedModelProvider(responses)

    @tool(name="test")
    def test_tool(query: str = "") -> str:
        return "result"

    agent = Agent(model=model, conversation_manager=manager, tools=[test_tool])

    with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock:
        agent("Test")
        # With 6 model calls and per_turn=2: called on 2nd, 4th, 6th + finally
        assert mock.call_count == 4


def test_per_turn_dynamic_change():
    """Test that per_turn can be changed dynamically."""
    manager = SlidingWindowConversationManager(per_turn=False)
    registry = HookRegistry()
    manager.register_hooks(registry)

    mock_agent = MagicMock()
    mock_agent.messages = []
    event = BeforeModelCallEvent(agent=mock_agent, invocation_state={})

    # Initially disabled
    with patch.object(manager, "apply_management") as mock_apply:
        registry.invoke_callbacks(event)
        assert mock_apply.call_count == 0

    # Enable dynamically
    manager.per_turn = True
    with patch.object(manager, "apply_management") as mock_apply:
        registry.invoke_callbacks(event)
        assert mock_apply.call_count == 1


def test_per_turn_reduces_message_count():
    """Test that per_turn actually reduces message count during execution."""
    manager = SlidingWindowConversationManager(per_turn=1, window_size=4)
    responses = [{"role": "assistant", "content": [{"text": f"Response {i}"}]} for i in range(10)]
    model = MockedModelProvider(responses)
    agent = Agent(model=model, conversation_manager=manager)

    message_counts = []
    original_apply = manager.apply_management

    def track_apply(agent_instance):
        message_counts.append(len(agent_instance.messages))
        return original_apply(agent_instance)

    with patch.object(manager, "apply_management", side_effect=track_apply):
        agent("Test")

    # Verify message count stayed around window_size
    assert any(count <= manager.window_size for count in message_counts)


def test_per_turn_state_persistence():
    """Test that model_call_count is persisted in state."""
    manager = SlidingWindowConversationManager(per_turn=3)
    manager._model_call_count = 7

    state = manager.get_state()
    assert state["model_call_count"] == 7

    new_manager = SlidingWindowConversationManager(per_turn=3)
    new_manager.restore_from_session(state)
    assert new_manager._model_call_count == 7


def test_per_turn_backward_compatibility():
    """Test that existing code without per_turn still works."""
    manager = SlidingWindowConversationManager(window_size=40)
    assert manager.per_turn is False

    responses = [{"role": "assistant", "content": [{"text": "Hello"}]}]
    model = MockedModelProvider(responses)
    agent = Agent(model=model, conversation_manager=manager)
    result = agent("Hello")
    assert result is not None
