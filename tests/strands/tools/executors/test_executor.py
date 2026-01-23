import unittest.mock
from unittest.mock import MagicMock

import pytest

import strands
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.telemetry.metrics import Trace
from strands.tools.executors._executor import ToolExecutor
from strands.types._events import ToolCancelEvent, ToolInterruptEvent, ToolResultEvent, ToolStreamEvent
from strands.types.tools import ToolUse


@pytest.fixture
def executor_cls():
    class ClsExecutor(ToolExecutor):
        def _execute(self, _agent, _tool_uses, _tool_results, _invocation_state):
            raise NotImplementedError

    return ClsExecutor


@pytest.fixture
def executor(executor_cls):
    return executor_cls()


@pytest.fixture
def tracer():
    with unittest.mock.patch.object(strands.tools.executors._executor, "get_tracer") as mock_get_tracer:
        yield mock_get_tracer.return_value


@pytest.mark.asyncio
async def test_executor_stream_yields_result(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_events = hook_events
    exp_hook_events = [
        BeforeToolCallEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
        ),
        AfterToolCallEvent(
            agent=agent,
            selected_tool=weather_tool,
            tool_use=tool_use,
            invocation_state=invocation_state,
            result=exp_results[0],
        ),
    ]
    assert tru_hook_events == exp_hook_events


@pytest.mark.asyncio
async def test_executor_stream_wraps_results(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    weather_tool.stream.return_value = agenerator(
        ["value 1", {"nested": True}, {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}]
    )

    tru_events = await alist(stream)
    exp_events = [
        ToolStreamEvent(tool_use, "value 1"),
        ToolStreamEvent(tool_use, {"nested": True}),
        ToolStreamEvent(tool_use, {"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_passes_through_typed_events(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    event_1 = ToolStreamEvent(tool_use, "value 1")
    event_2 = ToolStreamEvent(tool_use, {"nested": True})
    event_3 = ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]})
    weather_tool.stream.return_value = agenerator(
        [
            event_1,
            event_2,
            event_3,
        ]
    )

    tru_events = await alist(stream)
    assert tru_events[0] is event_1
    assert tru_events[1] is event_2

    # ToolResults are not passed through directly, they're unwrapped then wraped again
    assert tru_events[2] == event_3


@pytest.mark.asyncio
async def test_executor_stream_wraps_stream_events_if_no_result(
    executor, agent, tool_results, invocation_state, hook_events, weather_tool, alist, agenerator
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    weather_tool.stream = MagicMock()
    last_event = ToolStreamEvent(tool_use, "value 1")
    # Only ToolResultEvent can be the last value; all others are wrapped in ToolResultEvent
    weather_tool.stream.return_value = agenerator(
        [
            last_event,
        ]
    )

    tru_events = await alist(stream)
    exp_events = [last_event, ToolResultEvent(last_event)]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_yields_tool_error(
    executor, agent, tool_results, invocation_state, hook_events, exception_tool, alist
):
    tool_use = {"name": "exception_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [ToolResultEvent({"toolUseId": "1", "status": "error", "content": [{"text": "Error: Tool error"}]})]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolCallEvent(
        agent=agent,
        selected_tool=exception_tool,
        tool_use=tool_use,
        invocation_state=invocation_state,
        result=exp_results[0],
        exception=unittest.mock.ANY,
    )
    assert tru_hook_after_event == exp_hook_after_event


@pytest.mark.asyncio
async def test_executor_stream_yields_unknown_tool(executor, agent, tool_results, invocation_state, hook_events, alist):
    tool_use = {"name": "unknown_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "error", "content": [{"text": "Unknown tool: unknown_tool"}]})
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_hook_after_event = hook_events[-1]
    exp_hook_after_event = AfterToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use=tool_use,
        invocation_state=invocation_state,
        result=exp_results[0],
    )
    assert tru_hook_after_event == exp_hook_after_event


@pytest.mark.asyncio
async def test_executor_stream_with_trace(
    executor, tracer, agent, tool_results, cycle_trace, cycle_span, invocation_state, alist
):
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream_with_trace(agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "sunny"}]}),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tracer.start_tool_call_span.assert_called_once_with(
        tool_use, cycle_span, custom_trace_attributes=agent.trace_attributes
    )
    tracer.end_tool_call_span.assert_called_once_with(
        tracer.start_tool_call_span.return_value,
        {"content": [{"text": "sunny"}], "status": "success", "toolUseId": "1"},
    )

    cycle_trace.add_child.assert_called_once()
    assert isinstance(cycle_trace.add_child.call_args[0][0], Trace)


@pytest.mark.parametrize(
    ("cancel_tool", "cancel_message"),
    [(True, "tool cancelled by user"), ("user cancel message", "user cancel message")],
)
@pytest.mark.asyncio
async def test_executor_stream_cancel(
    cancel_tool, cancel_message, executor, agent, tool_results, invocation_state, alist
):
    def cancel_callback(event):
        event.cancel_tool = cancel_tool
        return event

    agent.hooks.add_callback(BeforeToolCallEvent, cancel_callback)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolCancelEvent(tool_use, cancel_message),
        ToolResultEvent(
            {
                "toolUseId": "1",
                "status": "error",
                "content": [{"text": cancel_message}],
            },
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_sets_span_attributes(
    executor, agent, tool_results, invocation_state, weather_tool, alist
):
    """Test that span attributes are set correctly when tool_spec is available."""
    with unittest.mock.patch("strands.tools.executors._executor.trace_api") as mock_trace_api:
        mock_span = unittest.mock.MagicMock()
        mock_trace_api.get_current_span.return_value = mock_span

        # Mock tool_spec with inputSchema containing json field
        with unittest.mock.patch.object(
            type(weather_tool), "tool_spec", new_callable=unittest.mock.PropertyMock
        ) as mock_tool_spec:
            mock_tool_spec.return_value = {
                "name": "weather_tool",
                "description": "Get weather information",
                "inputSchema": {"json": {"type": "object", "properties": {}}, "type": "object"},
            }

            tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
            stream = executor._stream(agent, tool_use, tool_results, invocation_state)

            await alist(stream)

            # Verify set_attribute was called with correct values
            calls = mock_span.set_attribute.call_args_list
            assert len(calls) == 2

            # Check description attribute
            assert calls[0][0][0] == "gen_ai.tool.description"
            assert calls[0][0][1] == "Get weather information"

            # Check json_schema attribute
            assert calls[1][0][0] == "gen_ai.tool.json_schema"
            # The serialize function should have been called on the json field


@pytest.mark.asyncio
async def test_executor_stream_handles_missing_json_in_input_schema(
    executor, agent, tool_results, invocation_state, weather_tool, alist
):
    """Test that span attributes handle inputSchema without json field gracefully."""
    with unittest.mock.patch("strands.tools.executors._executor.trace_api") as mock_trace_api:
        mock_span = unittest.mock.MagicMock()
        mock_trace_api.get_current_span.return_value = mock_span

        # Mock tool_spec with inputSchema but no json field
        with unittest.mock.patch.object(
            type(weather_tool), "tool_spec", new_callable=unittest.mock.PropertyMock
        ) as mock_tool_spec:
            mock_tool_spec.return_value = {
                "name": "weather_tool",
                "description": "Get weather information",
                "inputSchema": {"type": "object", "properties": {}},
            }

            tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}
            stream = executor._stream(agent, tool_use, tool_results, invocation_state)

            # Should not raise an error - json_schema attribute just won't be set
            await alist(stream)

            # Verify only description attribute was set (not json_schema)
            calls = mock_span.set_attribute.call_args_list
            assert len(calls) == 1
            assert calls[0][0][0] == "gen_ai.tool.description"


@pytest.mark.asyncio
async def test_executor_stream_no_span_attributes_when_no_tool_spec(
    executor, agent, tool_results, invocation_state, alist
):
    """Test that no span attributes are set when tool_spec is None."""
    with unittest.mock.patch("strands.tools.executors._executor.trace_api") as mock_trace_api:
        mock_span = unittest.mock.MagicMock()
        mock_trace_api.get_current_span.return_value = mock_span

        # Use unknown tool which will have no tool_spec
        tool_use: ToolUse = {"name": "unknown_tool", "toolUseId": "1", "input": {}}
        stream = executor._stream(agent, tool_use, tool_results, invocation_state)

        await alist(stream)

        # Verify set_attribute was not called since tool_spec is None
        mock_span.set_attribute.assert_not_called()


@pytest.mark.asyncio
async def test_executor_stream_hook_interrupt(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "weather_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:before_tool_call:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
    )

    def interrupt_callback(event):
        event.interrupt("test_name", reason="test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [ToolInterruptEvent(tool_use, [interrupt])]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = []
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_hook_interrupt_resume(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "weather_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:before_tool_call:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
        response="test response",
    )
    agent._interrupt_state.interrupts[interrupt.id] = interrupt

    interrupt_response = {}

    def interrupt_callback(event):
        interrupt_response["response"] = event.interrupt("test_name", reason="test reason")

    agent.hooks.add_callback(BeforeToolCallEvent, interrupt_callback)

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent(
            {
                "toolUseId": "test_tool_id",
                "status": "success",
                "content": [{"text": "sunny"}],
            },
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results

    tru_response = interrupt_response["response"]
    exp_response = "test response"
    assert tru_response == exp_response


@pytest.mark.asyncio
async def test_executor_stream_tool_interrupt(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "interrupt_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:tool_call:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
    )

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [ToolInterruptEvent(tool_use, [interrupt])]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = []
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_tool_interrupt_resume(executor, agent, tool_results, invocation_state, alist):
    tool_use = {"name": "interrupt_tool", "toolUseId": "test_tool_id", "input": {}}

    interrupt = Interrupt(
        id="v1:tool_call:test_tool_id:78714d6c-613c-5cf4-bf25-7037569941f9",
        name="test_name",
        reason="test reason",
        response="test response",
    )
    agent._interrupt_state.interrupts[interrupt.id] = interrupt

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)
    exp_events = [
        ToolResultEvent(
            {
                "toolUseId": "test_tool_id",
                "status": "success",
                "content": [{"text": "test response"}],
            },
        ),
    ]
    assert tru_events == exp_events

    tru_results = tool_results
    exp_results = [exp_events[-1].tool_result]
    assert tru_results == exp_results


@pytest.mark.asyncio
async def test_executor_stream_updates_invocation_state_with_agent(
    executor, agent, tool_results, invocation_state, weather_tool, alist
):
    """Test that invocation_state is updated with agent reference."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    # Start with empty invocation_state to verify agent is added
    empty_invocation_state = {}

    stream = executor._stream(agent, tool_use, tool_results, empty_invocation_state)
    await alist(stream)

    # Verify that the invocation_state was updated with the agent
    assert "agent" in empty_invocation_state
    assert empty_invocation_state["agent"] is agent


@pytest.mark.asyncio
async def test_executor_stream_retry_on_success(executor, agent, tool_results, invocation_state, alist):
    """Test that setting retry=True on AfterToolCallEvent causes tool re-execution."""
    call_count = {"count": 0}

    @strands.tool(name="retry_tool")
    def retry_tool():
        call_count["count"] += 1
        return f"attempt_{call_count['count']}"

    agent.tool_registry.register_tool(retry_tool)

    # Retry once, then succeed
    def retry_callback(event):
        if isinstance(event, AfterToolCallEvent) and call_count["count"] < 2:
            event.retry = True
        return event

    agent.hooks.add_callback(AfterToolCallEvent, retry_callback)

    tool_use: ToolUse = {"name": "retry_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)

    # Should have been called twice due to retry
    assert call_count["count"] == 2

    # Final result should be from second attempt
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "attempt_2"}]}),
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_retry_on_error(executor, agent, tool_results, invocation_state, alist):
    """Test that retry works when tool raises an exception."""
    call_count = {"count": 0}

    @strands.tool(name="error_retry_tool")
    def error_retry_tool():
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise RuntimeError("Simulated failure")
        return "success"

    agent.tool_registry.register_tool(error_retry_tool)

    # Retry on error - check result status since @tool decorator catches exceptions
    def retry_callback(event):
        if isinstance(event, AfterToolCallEvent) and event.result.get("status") == "error":
            event.retry = True
        return event

    agent.hooks.add_callback(AfterToolCallEvent, retry_callback)

    tool_use: ToolUse = {"name": "error_retry_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)

    # Should have been called twice - first failed, second succeeded
    assert call_count["count"] == 2

    # Final result should be success
    exp_events = [
        ToolResultEvent({"toolUseId": "1", "status": "success", "content": [{"text": "success"}]}),
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_executor_stream_retry_respects_max_retries(executor, agent, tool_results, invocation_state, alist):
    """Test that retry can be limited by the hook callback."""
    call_count = {"count": 0}
    max_retries = 3

    @strands.tool(name="limited_retry_tool")
    def limited_retry_tool():
        call_count["count"] += 1
        raise RuntimeError(f"Failure {call_count['count']}")

    agent.tool_registry.register_tool(limited_retry_tool)

    # Retry up to max_retries times - check result status since @tool decorator catches exceptions
    def retry_callback(event):
        if isinstance(event, AfterToolCallEvent) and event.result.get("status") == "error":
            if call_count["count"] < max_retries:
                event.retry = True
        return event

    agent.hooks.add_callback(AfterToolCallEvent, retry_callback)

    tool_use: ToolUse = {"name": "limited_retry_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)

    # Should have been called max_retries times
    assert call_count["count"] == max_retries

    # Final result should be error from last attempt
    assert len(tru_events) == 1
    assert tru_events[0].tool_result["status"] == "error"
    assert "Failure 3" in tru_events[0].tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_executor_stream_retry_on_exception(executor, agent, tool_results, invocation_state, alist):
    """Test that retry works when tool.stream raises an exception directly."""
    call_count = {"count": 0}

    @strands.tool(name="stream_exception_tool")
    def stream_exception_tool():
        return "success"

    # Replace stream to raise exception directly (like real exception_tool fixture)
    async def mock_stream(_tool_use, _invocation_state, **kwargs):
        call_count["count"] += 1
        if call_count["count"] < 2:
            raise RuntimeError("Stream exception")
        yield {"toolUseId": "1", "status": "success", "content": [{"text": "recovered"}]}

    stream_exception_tool.stream = mock_stream
    agent.tool_registry.register_tool(stream_exception_tool)

    # Retry on exception - event.exception is set when executor catches exception
    def retry_callback(event):
        if isinstance(event, AfterToolCallEvent) and event.exception is not None:
            event.retry = True
        return event

    agent.hooks.add_callback(AfterToolCallEvent, retry_callback)

    tool_use: ToolUse = {"name": "stream_exception_tool", "toolUseId": "1", "input": {}}
    stream = executor._stream(agent, tool_use, tool_results, invocation_state)

    tru_events = await alist(stream)

    # Should have been called twice - first raised exception, second succeeded
    assert call_count["count"] == 2

    # Final result should be success (last event is ToolResultEvent)
    assert tru_events[-1].tool_result["status"] == "success"
    assert "recovered" in tru_events[-1].tool_result["content"][0]["text"]


@pytest.mark.asyncio
async def test_executor_stream_retry_field_is_writable(executor, agent, tool_results, invocation_state, alist):
    """Test that the retry field on AfterToolCallEvent can be written."""
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "1", "input": {}}

    # Verify retry field can be set
    def check_retry_writable(event):
        if isinstance(event, AfterToolCallEvent):
            assert event._can_write("retry") is True
            # Don't actually retry, just verify it's writable
        return event

    agent.hooks.add_callback(AfterToolCallEvent, check_retry_writable)

    stream = executor._stream(agent, tool_use, tool_results, invocation_state)
    await alist(stream)
