"""Tests for AfterToolCallEvent interrupt support.

Covers all 4 interrupt paths in _executor.py:
- Success path: tool succeeds, after-hook interrupts
- Exception path: tool raises, after-hook interrupts
- Cancel path: before-hook cancels tool, after-hook interrupts
- Unknown tool path: tool not found, after-hook interrupts

Also covers resume behavior and retry-on-resume.
"""

import pytest

import strands
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.tools.executors._executor import ToolExecutor
from strands.types._events import ToolCancelEvent, ToolInterruptEvent, ToolResultEvent
from strands.types.tools import ToolUse


@pytest.fixture
def executor():
    class TestExecutor(ToolExecutor):
        def _execute(self, _agent, _tool_uses, _tool_results, _invocation_state):
            raise NotImplementedError

    return TestExecutor()


# -- Success path --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_on_success(executor, agent, tool_results, invocation_state, alist):
    """AfterToolCallEvent interrupt on successful tool execution yields ToolInterruptEvent with source_event."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "t1", "input": {}}

    def interrupt_after(event):
        if isinstance(event, AfterToolCallEvent):
            event.interrupt("review", reason="check result")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after)

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert len(events) == 1
    assert isinstance(events[0], ToolInterruptEvent)
    assert events[0].interrupts[0].name == "review"
    assert isinstance(events[0].source_event, AfterToolCallEvent)
    assert events[0].source_event.result["status"] == "success"

    # Result preserved in tool_results for resume
    assert len(tool_results) == 1
    assert tool_results[0]["toolUseId"] == "t1"
    assert tool_results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_resume_on_success(executor, agent, tool_results, invocation_state, alist):
    """On resume, after-hook re-fires and callback gets the interrupt response."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "t1", "input": {}}

    interrupt = Interrupt(
        id="v1:after_tool_call:t1:fd6381ef-9533-5ce1-8a4d-75db796edf35",
        name="review",
        reason="check result",
        response="APPROVED",
    )
    agent._interrupt_state.interrupts[interrupt.id] = interrupt

    captured = {}

    def interrupt_after(event):
        if isinstance(event, AfterToolCallEvent):
            captured["response"] = event.interrupt("review", reason="check result")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after)

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    # No interrupt this time — response was available
    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert captured["response"] == "APPROVED"


# -- Exception path --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_on_exception(executor, agent, tool_results, invocation_state, alist):
    """AfterToolCallEvent interrupt when tool raises an exception."""
    tool_use: ToolUse = {"name": "exception_tool", "toolUseId": "t1", "input": {}}

    def interrupt_on_error(event):
        if isinstance(event, AfterToolCallEvent) and event.exception:
            event.interrupt("error_review", reason=str(event.exception))

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_on_error)

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert len(events) == 1
    assert isinstance(events[0], ToolInterruptEvent)
    assert events[0].interrupts[0].name == "error_review"
    assert isinstance(events[0].source_event, AfterToolCallEvent)
    assert events[0].source_event.result["status"] == "error"

    # Result preserved
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "error"


# -- Cancel path --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_on_cancel(executor, agent, tool_results, invocation_state, alist):
    """AfterToolCallEvent interrupt when tool was cancelled by before-hook."""

    def cancel_tool(event):
        if isinstance(event, BeforeToolCallEvent):
            event.cancel_tool = True

    def interrupt_after_cancel(event):
        if isinstance(event, AfterToolCallEvent) and event.cancel_message:
            event.interrupt("cancel_review", reason="cancelled")

    agent.hooks.add_callback(BeforeToolCallEvent, cancel_tool)
    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after_cancel)

    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "t1", "input": {}}
    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    # ToolCancelEvent then ToolInterruptEvent
    assert isinstance(events[0], ToolCancelEvent)
    assert isinstance(events[1], ToolInterruptEvent)
    assert events[1].interrupts[0].name == "cancel_review"
    assert isinstance(events[1].source_event, AfterToolCallEvent)

    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "error"


# -- Unknown tool path --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_on_unknown_tool(executor, agent, tool_results, invocation_state, alist):
    """AfterToolCallEvent interrupt when tool is not found."""

    def interrupt_on_unknown(event):
        if isinstance(event, AfterToolCallEvent) and event.exception:
            event.interrupt("unknown_review", reason="tool missing")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_on_unknown)

    tool_use: ToolUse = {"name": "nonexistent_tool", "toolUseId": "t1", "input": {}}
    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert len(events) == 1
    assert isinstance(events[0], ToolInterruptEvent)
    assert events[0].interrupts[0].name == "unknown_review"

    assert len(tool_results) == 1
    assert "Unknown tool" in tool_results[0]["content"][0]["text"]


# -- Interrupt ID uniqueness --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_id_uses_tool_use_id(executor, agent, tool_results, invocation_state, alist):
    """Interrupt ID includes toolUseId so different tool calls produce different IDs."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "unique_id_123", "input": {}}

    def interrupt_after(event):
        if isinstance(event, AfterToolCallEvent):
            event.interrupt("check", reason="test")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after)

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    interrupt = events[0].interrupts[0]
    assert "unique_id_123" in interrupt.id
    assert interrupt.id.startswith("v1:after_tool_call:")


# -- source_event typing --


@pytest.mark.asyncio
async def test_tool_interrupt_event_source_event_none_for_before_hook(
    executor, agent, tool_results, invocation_state, alist
):
    """BeforeToolCallEvent interrupts produce ToolInterruptEvent with source_event=None."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "t1", "input": {}}

    def interrupt_before(event):
        if isinstance(event, BeforeToolCallEvent):
            event.interrupt("before_check", reason="test")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_before)

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert isinstance(events[0], ToolInterruptEvent)
    assert events[0].source_event is None


# -- No interrupt when not raised --


@pytest.mark.asyncio
async def test_after_tool_call_no_interrupt_when_not_raised(executor, agent, tool_results, invocation_state, alist):
    """Normal after-hook without interrupt proceeds as usual."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "t1", "input": {}}

    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_result["status"] == "success"


# -- Interrupt takes precedence over retry --


@pytest.mark.asyncio
async def test_after_tool_call_interrupt_takes_precedence_over_retry(
    executor, agent, tool_results, invocation_state, alist
):
    """When interrupt is raised, retry flag on the event is never checked."""
    call_count = {"n": 0}

    @strands.tool(name="counted_tool")
    def counted_tool():
        call_count["n"] += 1
        return "done"

    agent.tool_registry.register_tool(counted_tool)

    def interrupt_and_retry(event):
        if isinstance(event, AfterToolCallEvent):
            # interrupt() raises InterruptException before retry is read
            event.retry = True
            event.interrupt("block", reason="paused")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_and_retry)

    tool_use: ToolUse = {"name": "counted_tool", "toolUseId": "t1", "input": {}}
    events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    # Tool ran once, interrupt stopped the loop
    assert call_count["n"] == 1
    assert isinstance(events[0], ToolInterruptEvent)
