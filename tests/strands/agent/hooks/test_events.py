from unittest.mock import Mock

import pytest

from strands.agent.agent_result import AgentResult
from strands.hooks import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.types.content import Message, Messages
from strands.types.tools import ToolResult, ToolUse


@pytest.fixture
def agent():
    return Mock()


@pytest.fixture
def sample_messages() -> Messages:
    return [{"role": "user", "content": [{"text": "Hello, agent!"}]}]


@pytest.fixture
def tool():
    tool = Mock()
    tool.tool_name = "test_tool"
    return tool


@pytest.fixture
def tool_use():
    return ToolUse(name="test_tool", toolUseId="123", input={"param": "value"})


@pytest.fixture
def tool_invocation_state():
    return {"param": "value"}


@pytest.fixture
def tool_result():
    return ToolResult(content=[{"text": "result"}], status="success", toolUseId="123")


@pytest.fixture
def initialized_event(agent):
    return AgentInitializedEvent(agent=agent)


@pytest.fixture
def start_request_event(agent):
    return BeforeInvocationEvent(agent=agent)


@pytest.fixture
def start_request_event_with_messages(agent, sample_messages):
    return BeforeInvocationEvent(agent=agent, messages=sample_messages)


@pytest.fixture
def messaged_added_event(agent):
    return MessageAddedEvent(agent=agent, message=Mock())


@pytest.fixture
def end_request_event(agent):
    return AfterInvocationEvent(agent=agent)


@pytest.fixture
def before_tool_event(agent, tool, tool_use, tool_invocation_state):
    return BeforeToolCallEvent(
        agent=agent,
        selected_tool=tool,
        tool_use=tool_use,
        invocation_state=tool_invocation_state,
    )


@pytest.fixture
def after_tool_event(agent, tool, tool_use, tool_invocation_state, tool_result):
    return AfterToolCallEvent(
        agent=agent,
        selected_tool=tool,
        tool_use=tool_use,
        invocation_state=tool_invocation_state,
        result=tool_result,
    )


def test_event_should_reverse_callbacks(
    initialized_event,
    start_request_event,
    messaged_added_event,
    end_request_event,
    before_tool_event,
    after_tool_event,
):
    # note that we ignore E712 (explicit booleans) for consistency/readability purposes

    assert initialized_event.should_reverse_callbacks == False  # noqa: E712

    assert messaged_added_event.should_reverse_callbacks == False  # noqa: E712

    assert start_request_event.should_reverse_callbacks == False  # noqa: E712
    assert end_request_event.should_reverse_callbacks == True  # noqa: E712

    assert before_tool_event.should_reverse_callbacks == False  # noqa: E712
    assert after_tool_event.should_reverse_callbacks == True  # noqa: E712


def test_message_added_event_cannot_write_properties(messaged_added_event):
    with pytest.raises(AttributeError, match="Property agent is not writable"):
        messaged_added_event.agent = Mock()
    with pytest.raises(AttributeError, match="Property message is not writable"):
        messaged_added_event.message = {}


def test_before_tool_invocation_event_can_write_properties(before_tool_event):
    new_tool_use = ToolUse(name="new_tool", toolUseId="456", input={})
    before_tool_event.selected_tool = None  # Should not raise
    before_tool_event.tool_use = new_tool_use  # Should not raise


def test_before_tool_invocation_event_cannot_write_properties(before_tool_event):
    with pytest.raises(AttributeError, match="Property agent is not writable"):
        before_tool_event.agent = Mock()
    with pytest.raises(AttributeError, match="Property invocation_state is not writable"):
        before_tool_event.invocation_state = {}


def test_after_tool_invocation_event_can_write_properties(after_tool_event):
    new_result = ToolResult(content=[{"text": "new result"}], status="success", toolUseId="456")
    after_tool_event.result = new_result  # Should not raise


def test_after_tool_invocation_event_cannot_write_properties(after_tool_event):
    with pytest.raises(AttributeError, match="Property agent is not writable"):
        after_tool_event.agent = Mock()
    with pytest.raises(AttributeError, match="Property selected_tool is not writable"):
        after_tool_event.selected_tool = None
    with pytest.raises(AttributeError, match="Property tool_use is not writable"):
        after_tool_event.tool_use = ToolUse(name="new", toolUseId="456", input={})
    with pytest.raises(AttributeError, match="Property invocation_state is not writable"):
        after_tool_event.invocation_state = {}
    with pytest.raises(AttributeError, match="Property exception is not writable"):
        after_tool_event.exception = Exception("test")


def test_after_invocation_event_properties_not_writable(agent):
    """Test that properties are not writable after initialization."""
    mock_message: Message = {"role": "assistant", "content": [{"text": "test"}]}
    mock_result = AgentResult(
        stop_reason="end_turn",
        message=mock_message,
        metrics={},
        state={},
    )

    event = AfterInvocationEvent(agent=agent, result=None)

    with pytest.raises(AttributeError, match="Property result is not writable"):
        event.result = mock_result

    with pytest.raises(AttributeError, match="Property agent is not writable"):
        event.agent = Mock()


def test_before_invocation_event_messages_default_none(agent):
    """Test that BeforeInvocationEvent.messages defaults to None for backward compatibility."""
    event = BeforeInvocationEvent(agent=agent)
    assert event.messages is None


def test_before_invocation_event_messages_writable(agent, sample_messages):
    """Test that BeforeInvocationEvent.messages can be modified in-place for guardrail redaction."""
    event = BeforeInvocationEvent(agent=agent, messages=sample_messages)

    # Should be able to modify the messages list in-place
    event.messages[0]["content"] = [{"text": "[REDACTED]"}]
    assert event.messages[0]["content"] == [{"text": "[REDACTED]"}]

    # Should be able to reassign messages entirely
    new_messages: Messages = [{"role": "user", "content": [{"text": "Different message"}]}]
    event.messages = new_messages
    assert event.messages == new_messages


def test_before_invocation_event_agent_not_writable(start_request_event_with_messages):
    """Test that BeforeInvocationEvent.agent is not writable."""
    with pytest.raises(AttributeError, match="Property agent is not writable"):
        start_request_event_with_messages.agent = Mock()


# Tests for BeforeContextReductionEvent and AfterContextReductionEvent


@pytest.fixture
def context_overflow_exception():
    from strands.types.exceptions import ContextWindowOverflowException

    return ContextWindowOverflowException("Context window exceeded")


@pytest.fixture
def before_context_reduction_event(agent, context_overflow_exception):
    from strands.hooks import BeforeContextReductionEvent

    return BeforeContextReductionEvent(
        agent=agent,
        exception=context_overflow_exception,
        message_count=50,
    )


@pytest.fixture
def after_context_reduction_event(agent):
    from strands.hooks import AfterContextReductionEvent

    return AfterContextReductionEvent(
        agent=agent,
        original_message_count=50,
        new_message_count=25,
        removed_count=25,
    )


@pytest.fixture
def after_context_reduction_event_with_exception(agent):
    from strands.hooks import AfterContextReductionEvent

    return AfterContextReductionEvent(
        agent=agent,
        original_message_count=50,
        new_message_count=50,
        removed_count=0,
        exception=RuntimeError("Reduction failed"),
    )


def test_before_context_reduction_event_properties(before_context_reduction_event, context_overflow_exception):
    assert before_context_reduction_event.exception == context_overflow_exception
    assert before_context_reduction_event.message_count == 50
    assert before_context_reduction_event.should_reverse_callbacks is False


def test_after_context_reduction_event_properties(after_context_reduction_event):
    assert after_context_reduction_event.original_message_count == 50
    assert after_context_reduction_event.new_message_count == 25
    assert after_context_reduction_event.removed_count == 25
    assert after_context_reduction_event.exception is None
    assert after_context_reduction_event.should_reverse_callbacks is True


def test_after_context_reduction_event_with_exception(after_context_reduction_event_with_exception):
    assert after_context_reduction_event_with_exception.original_message_count == 50
    assert after_context_reduction_event_with_exception.new_message_count == 50
    assert after_context_reduction_event_with_exception.removed_count == 0
    assert after_context_reduction_event_with_exception.exception is not None
    assert isinstance(after_context_reduction_event_with_exception.exception, RuntimeError)
