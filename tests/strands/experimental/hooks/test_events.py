from unittest.mock import Mock

import pytest

from strands.experimental.hooks.events import (
    AfterToolInvocationEvent,
    AgentInitializedEvent,
    BeforeToolInvocationEvent,
    EndRequestEvent,
    StartRequestEvent,
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
def tool_kwargs():
    return {"param": "value"}


@pytest.fixture
def tool_result():
    return ToolResult(content=[{"text": "result"}], status="success", toolUseId="123")


@pytest.fixture
def initialized_event(agent):
    return AgentInitializedEvent(agent=agent)


@pytest.fixture
def start_request_event(agent):
    return StartRequestEvent(agent=agent)


@pytest.fixture
def end_request_event(agent):
    return EndRequestEvent(agent=agent)


@pytest.fixture
def before_tool_event(agent, tool, tool_use, tool_kwargs):
    return BeforeToolInvocationEvent(
        agent=agent,
        selected_tool=tool,
        tool_use=tool_use,
        kwargs=tool_kwargs,
    )


@pytest.fixture
def after_tool_event(agent, tool, tool_use, tool_kwargs, tool_result):
    return AfterToolInvocationEvent(
        agent=agent,
        selected_tool=tool,
        tool_use=tool_use,
        kwargs=tool_kwargs,
        result=tool_result,
    )


def test_event_should_reverse_callbacks(
    initialized_event,
    start_request_event,
    end_request_event,
    before_tool_event,
    after_tool_event,
):
    # note that we ignore E712 (explicit booleans) for consistency/readability purposes

    assert initialized_event.should_reverse_callbacks == False  # noqa: E712

    assert start_request_event.should_reverse_callbacks == False  # noqa: E712
    assert end_request_event.should_reverse_callbacks == True  # noqa: E712

    assert before_tool_event.should_reverse_callbacks == False  # noqa: E712
    assert after_tool_event.should_reverse_callbacks == True  # noqa: E712


def test_before_tool_invocation_event_can_write_properties(before_tool_event):
    new_tool_use = ToolUse(name="new_tool", toolUseId="456", input={})
    before_tool_event.selected_tool = None  # Should not raise
    before_tool_event.tool_use = new_tool_use  # Should not raise


def test_before_tool_invocation_event_cannot_write_properties(before_tool_event):
    with pytest.raises(AttributeError, match="Property agent is not writable"):
        before_tool_event.agent = Mock()
    with pytest.raises(AttributeError, match="Property kwargs is not writable"):
        before_tool_event.kwargs = {}


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
    with pytest.raises(AttributeError, match="Property kwargs is not writable"):
        after_tool_event.kwargs = {}
    with pytest.raises(AttributeError, match="Property exception is not writable"):
        after_tool_event.exception = Exception("test")
