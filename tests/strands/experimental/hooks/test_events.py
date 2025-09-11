from unittest.mock import Mock

import pytest

from strands.experimental.hooks import AfterToolInvocationEvent, BeforeToolInvocationEvent
from strands.hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    MessageAddedEvent,
)
from strands.types.tools import ToolResult, ToolUse


@pytest.fixture
def agent():
    return Mock()


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
def messaged_added_event(agent):
    return MessageAddedEvent(agent=agent, message=Mock())


@pytest.fixture
def end_request_event(agent):
    return AfterInvocationEvent(agent=agent)


@pytest.fixture
def before_tool_event(agent, tool, tool_use, tool_invocation_state):
    return BeforeToolInvocationEvent(
        agent=agent,
        selected_tool=tool,
        tool_use=tool_use,
        invocation_state=tool_invocation_state,
    )


@pytest.fixture
def after_tool_event(agent, tool, tool_use, tool_invocation_state, tool_result):
    return AfterToolInvocationEvent(
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
