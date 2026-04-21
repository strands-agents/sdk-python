"""Tests for AfterToolCallEvent interrupt resume in the event loop.

Covers the replay logic in _handle_tool_execution:
- Resume replays after-hook, callback gets response
- Resume with retry=True re-queues tool for execution
- Resume with result modification preserves the modified result
- Multiple tools where one has after-interrupt
"""

import threading
import unittest.mock

import pytest

import strands
import strands.event_loop.event_loop
from strands import Agent
from strands.event_loop._retry import ModelRetryStrategy
from strands.hooks import AfterToolCallEvent, HookRegistry
from strands.interrupt import Interrupt, _InterruptState
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.executors import SequentialToolExecutor
from strands.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def tool(tool_registry):
    @strands.tool
    def tool_for_testing(random_string: str):
        return random_string

    tool_registry.register_tool(tool_for_testing)
    return tool_for_testing


@pytest.fixture
def hook_registry():
    registry = HookRegistry()
    retry_strategy = ModelRetryStrategy()
    retry_strategy.register_hooks(registry)
    return registry


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def agent(model, tool_registry, hook_registry):
    mock = unittest.mock.Mock(name="agent")
    mock.__class__ = Agent
    mock.config.cache_points = []
    mock.model = model
    mock.system_prompt = "test"
    mock.messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    mock.tool_registry = tool_registry
    mock.thread_pool = None
    mock.event_loop_metrics = EventLoopMetrics()
    mock.event_loop_metrics.reset_usage_metrics()
    mock.hooks = hook_registry
    mock.tool_executor = SequentialToolExecutor()
    mock._interrupt_state = _InterruptState()
    mock._cancel_signal = threading.Event()
    mock._model_state = {}
    mock.trace_attributes = {}
    mock.retry_strategy = ModelRetryStrategy()
    return mock


@pytest.mark.asyncio
async def test_after_tool_interrupt_and_resume(agent, model, tool, agenerator, alist):
    """Full cycle: tool runs → after-hook interrupts → resume → after-hook replays with response."""

    # Step 1: First invocation — tool runs, after-hook interrupts
    model.stream.return_value = agenerator(
        [
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "tool_for_testing"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "hello"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    def interrupt_after(event):
        if isinstance(event, AfterToolCallEvent):
            event.interrupt("approval", reason="needs review")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after)

    stream = strands.event_loop.event_loop.event_loop_cycle(agent, invocation_state={})
    events = await alist(stream)

    stop_reason = events[-1]["stop"][0]
    interrupts = events[-1]["stop"][4]
    assert stop_reason == "interrupt"
    assert len(interrupts) == 1
    assert interrupts[0].name == "approval"

    # Verify after_tool_events saved in context
    assert "after_tool_events" in agent._interrupt_state.context
    assert len(agent._interrupt_state.context["after_tool_events"]) == 1

    # Verify tool result preserved
    assert len(agent._interrupt_state.context["tool_results"]) == 1
    assert agent._interrupt_state.context["tool_results"][0]["status"] == "success"

    # Step 2: Resume — provide response, after-hook replays
    interrupt_id = interrupts[0].id
    agent._interrupt_state.interrupts[interrupt_id].response = "APPROVED"

    # Remove the interrupt hook and add one that captures the response
    captured = {}

    # Replace hook: on resume, interrupt() returns the response
    agent.hooks = HookRegistry()
    retry_strategy = ModelRetryStrategy()
    retry_strategy.register_hooks(agent.hooks)

    def capture_response(event):
        if isinstance(event, AfterToolCallEvent):
            captured["response"] = event.interrupt("approval", reason="needs review")

    agent.hooks.add_callback(AfterToolCallEvent, capture_response)

    model.stream.return_value = agenerator([{"contentBlockStop": {}}])

    stream = strands.event_loop.event_loop.event_loop_cycle(agent, invocation_state={})
    events = await alist(stream)

    stop_reason = events[-1]["stop"][0]
    assert stop_reason == "end_turn"
    assert captured["response"] == "APPROVED"

    # Interrupt state cleared
    assert not agent._interrupt_state.activated


@pytest.mark.asyncio
async def test_after_tool_interrupt_resume_with_retry(agent, model, tool, agenerator, alist):
    """On resume, if after-hook sets retry=True, tool re-executes."""
    tool_use = {"toolUseId": "t1", "name": "tool_for_testing", "input": {"random_string": "attempt1"}}
    tool_use_message = {
        "role": "assistant",
        "content": [{"toolUse": tool_use}],
    }

    # Set up interrupt state as if first invocation already happened
    original_result = {"toolUseId": "t1", "status": "error", "content": [{"text": "failed"}]}

    interrupt = Interrupt(
        id="v1:after_tool_call:t1:7eb5933b-ed83-5e65-84e6-fa22d85940c9",
        name="retry_check",
        reason="tool failed",
        response="RETRY",
    )

    agent._interrupt_state.context = {
        "tool_use_message": tool_use_message,
        "tool_results": [original_result],
        "after_tool_events": [{"tool_use": tool_use, "result": original_result, "cancel_message": None}],
    }
    agent._interrupt_state.interrupts[interrupt.id] = interrupt
    agent._interrupt_state.activate()

    # On resume, after-hook gets response and sets retry
    def retry_on_response(event):
        if isinstance(event, AfterToolCallEvent) and event.result.get("status") == "error":
            response = event.interrupt("retry_check", reason="tool failed")
            if response == "RETRY":
                event.retry = True

    agent.hooks.add_callback(AfterToolCallEvent, retry_on_response)

    # Model responds with end_turn after the retried tool completes
    model.stream.side_effect = [agenerator([{"contentBlockStop": {}}])]

    stream = strands.event_loop.event_loop.event_loop_cycle(agent, invocation_state={})
    events = await alist(stream)

    stop_reason = events[-1]["stop"][0]
    assert stop_reason == "end_turn"

    # The tool re-executed (retry removed the error result and re-queued),
    # so the final result message should contain a success result
    result_messages = [
        m
        for m in agent.messages
        if isinstance(m, dict) and m.get("role") == "user" and any("toolResult" in c for c in m.get("content", []))
    ]
    assert len(result_messages) > 0
    last_tool_result = result_messages[-1]["content"][0]["toolResult"]
    assert last_tool_result["status"] == "success"


@pytest.mark.asyncio
async def test_after_tool_interrupt_resume_modifies_result(agent, model, tool, agenerator, alist):
    """On resume, after-hook can modify the result without retry."""
    tool_use = {"toolUseId": "t1", "name": "tool_for_testing", "input": {"random_string": "test"}}
    tool_use_message = {
        "role": "assistant",
        "content": [{"toolUse": tool_use}],
    }

    original_result = {"toolUseId": "t1", "status": "error", "content": [{"text": "original error"}]}

    interrupt = Interrupt(
        id="v1:after_tool_call:t1:6124fc2a-cbe6-5805-84ac-5847c3fe6953",
        name="fix_result",
        reason="error",
        response="USE_DEFAULT",
    )

    agent._interrupt_state.context = {
        "tool_use_message": tool_use_message,
        "tool_results": [original_result],
        "after_tool_events": [{"tool_use": tool_use, "result": original_result, "cancel_message": None}],
    }
    agent._interrupt_state.interrupts[interrupt.id] = interrupt
    agent._interrupt_state.activate()

    def modify_on_response(event):
        if isinstance(event, AfterToolCallEvent):
            response = event.interrupt("fix_result", reason="error")
            if response == "USE_DEFAULT":
                event.result = {"toolUseId": "t1", "status": "success", "content": [{"text": "default value"}]}

    agent.hooks.add_callback(AfterToolCallEvent, modify_on_response)

    model.stream.return_value = agenerator([{"contentBlockStop": {}}])

    stream = strands.event_loop.event_loop.event_loop_cycle(agent, invocation_state={})
    events = await alist(stream)

    stop_reason = events[-1]["stop"][0]
    assert stop_reason == "end_turn"

    # The modified result should be in the conversation
    result_messages = [
        m
        for m in agent.messages
        if isinstance(m, dict) and m.get("role") == "user" and any("toolResult" in c for c in m.get("content", []))
    ]
    assert len(result_messages) > 0
    last_tool_result = result_messages[-1]["content"][0]["toolResult"]
    assert last_tool_result["status"] == "success"
    assert last_tool_result["content"][0]["text"] == "default value"


@pytest.mark.asyncio
async def test_after_tool_interrupt_context_saved_correctly(agent, model, tool, agenerator, alist):
    """Interrupt context includes after_tool_events list."""
    model.stream.return_value = agenerator(
        [
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "tool_for_testing"}}}},
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "x"}'}}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )

    def interrupt_after(event):
        if isinstance(event, AfterToolCallEvent):
            event.interrupt("check", reason="test")

    agent.hooks.add_callback(AfterToolCallEvent, interrupt_after)

    stream = strands.event_loop.event_loop.event_loop_cycle(agent, invocation_state={})
    await alist(stream)

    ctx = agent._interrupt_state.context
    assert "after_tool_events" in ctx
    assert len(ctx["after_tool_events"]) == 1
    snapshot = ctx["after_tool_events"][0]
    assert isinstance(snapshot, dict)
    assert "tool_use" in snapshot
    assert "result" in snapshot
    assert snapshot["result"]["toolUseId"] == "t1"
