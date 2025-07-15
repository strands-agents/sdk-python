import concurrent
import unittest.mock
from unittest.mock import ANY, MagicMock, call, patch

import pytest

import strands
import strands.telemetry
from strands.event_loop.event_loop import run_tool
from strands.experimental.hooks import (
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
)
from strands.hooks import (
    HookProvider,
    HookRegistry,
)
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException, ModelThrottledException
from tests.fixtures.mock_hook_provider import MockHookProvider


@pytest.fixture
def mock_time():
    with unittest.mock.patch.object(strands.event_loop.event_loop, "time") as mock:
        yield mock


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def system_prompt():
    return "p1"


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "Hello"}]}]


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def thread_pool():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def tool(tool_registry):
    @strands.tool
    def tool_for_testing(random_string: str):
        return random_string

    tool_registry.register_tool(tool_for_testing)

    return tool_for_testing


@pytest.fixture
def tool_times_2(tool_registry):
    @strands.tools.tool
    def multiply_by_2(x: int) -> int:
        return x * 2

    tool_registry.register_tool(multiply_by_2)

    return multiply_by_2


@pytest.fixture
def tool_times_5(tool_registry):
    @strands.tools.tool
    def multiply_by_5(x: int) -> int:
        return x * 5

    tool_registry.register_tool(multiply_by_5)

    return multiply_by_5


@pytest.fixture
def tool_stream(tool):
    return [
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": tool.tool_spec["name"],
                    },
                },
            },
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"random_string": "abcdEfghI123"}'}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]


@pytest.fixture
def hook_registry():
    return HookRegistry()


@pytest.fixture
def hook_provider(hook_registry):
    provider = MockHookProvider(
        event_types=[
            BeforeToolInvocationEvent,
            AfterToolInvocationEvent,
            BeforeModelInvocationEvent,
            AfterModelInvocationEvent,
        ]
    )
    hook_registry.add_hook(provider)
    return provider


@pytest.fixture
def agent(model, system_prompt, messages, tool_registry, thread_pool, hook_registry):
    mock = unittest.mock.Mock(name="agent")
    mock.config.cache_points = []
    mock.model = model
    mock.system_prompt = system_prompt
    mock.messages = messages
    mock.tool_registry = tool_registry
    mock.thread_pool = thread_pool
    mock.event_loop_metrics = EventLoopMetrics()
    mock.hooks = hook_registry

    return mock


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer.start_event_loop_cycle_span.return_value = MagicMock()
    tracer.start_model_invoke_span.return_value = MagicMock()
    return tracer


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response(
    agent,
    model,
    agenerator,
    alist,
):
    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_throttling(
    mock_time,
    agent,
    model,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state
    # Verify that sleep was called once with the initial delay
    mock_time.sleep.assert_called_once()


@pytest.mark.asyncio
async def test_event_loop_cycle_exponential_backoff(
    mock_time,
    agent,
    model,
    agenerator,
    alist,
):
    """Test that the exponential backoff works correctly with multiple retries."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    # Verify the final response
    assert tru_stop_reason == "end_turn"
    assert tru_message == {"role": "assistant", "content": [{"text": "test text"}]}
    assert tru_request_state == {}

    # Verify that sleep was called with increasing delays
    # Initial delay is 4, then 8, then 16
    assert mock_time.sleep.call_count == 3
    assert mock_time.sleep.call_args_list == [call(4), call(8), call(16)]


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_throttling_exceeded(
    mock_time,
    agent,
    model,
    alist,
):
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
    ]

    with pytest.raises(ModelThrottledException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    mock_time.sleep.assert_has_calls(
        [
            call(4),
            call(8),
            call(16),
            call(32),
            call(64),
        ]
    )


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response_error(
    agent,
    model,
    alist,
):
    model.stream.side_effect = RuntimeError("Unhandled error")

    with pytest.raises(RuntimeError):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result(
    agent,
    model,
    system_prompt,
    messages,
    tool_stream,
    tool_registry,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state

    model.stream.assert_called_with(
        [
            {"role": "user", "content": [{"text": "Hello"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "tool_for_testing",
                            "input": {"random_string": "abcdEfghI123"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "status": "success",
                            "content": [{"text": "abcdEfghI123"}],
                        },
                    },
                ],
            },
            {"role": "assistant", "content": [{"text": "test text"}]},
        ],
        tool_registry.get_all_tool_specs(),
        "p1",
    )


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_error(
    agent,
    model,
    tool_stream,
    agenerator,
    alist,
):
    model.stream.side_effect = [agenerator(tool_stream)]

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_no_tool_handler(
    agent,
    model,
    tool_stream,
    agenerator,
    alist,
):
    model.stream.side_effect = [agenerator(tool_stream)]
    # Set tool_handler to None for this test
    agent.tool_handler = None

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_stop(
    agent,
    model,
    tool,
    agenerator,
    alist,
):
    model.stream.side_effect = [
        agenerator(
            [
                {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": tool.tool_spec["name"],
                            },
                        },
                    },
                },
                {"contentBlockStop": {}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={"request_state": {"stop_event_loop": True}},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "tool_use"
    exp_message = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "input": {},
                    "name": "tool_for_testing",
                    "toolUseId": "t1",
                }
            }
        ],
    }
    exp_request_state = {"stop_event_loop": True}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state


@pytest.mark.asyncio
async def test_cycle_exception(
    agent,
    model,
    tool_stream,
    agenerator,
):
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(tool_stream),
        agenerator(tool_stream),
        ValueError("Invalid error presented"),
    ]

    tru_stop_event = None
    exp_stop_event = {"callback": {"force_stop": True, "force_stop_reason": "Invalid error presented"}}

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        async for event in stream:
            tru_stop_event = event

    assert tru_stop_event == exp_stop_event


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_cycle_creates_spans(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Call event_loop_cycle
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify tracer methods were called correctly
    mock_get_tracer.assert_called_once()
    mock_tracer.start_event_loop_cycle_span.assert_called_once()
    mock_tracer.start_model_invoke_span.assert_called_once()
    mock_tracer.end_model_invoke_span.assert_called_once()
    mock_tracer.end_event_loop_cycle_span.assert_called_once()


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_model_error(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to raise an exception
    model.stream.side_effect = ContextWindowOverflowException("Input too long")

    # Call event_loop_cycle, expecting it to handle the exception
    with pytest.raises(ContextWindowOverflowException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    # Verify error handling span methods were called
    mock_tracer.end_span_with_error.assert_called_once_with(model_span, "Input too long", model.stream.side_effect)


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_tool_execution(
    mock_get_tracer,
    agent,
    model,
    tool_stream,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to return tool use and then text response
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Call event_loop_cycle which should execute a tool
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify the parent_span parameter is passed to run_tools
    # At a minimum, verify both model spans were created (one for each model invocation)
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 2


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_throttling_exception(
    mock_get_tracer,
    agent,
    model,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to raise a throttling exception and then succeed
    model.stream.side_effect = [
        ModelThrottledException("Throttling Error"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Mock the time.sleep function to speed up the test
    with patch("strands.event_loop.event_loop.time.sleep"):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

    # Verify error span was created for the throttling exception
    assert mock_tracer.end_span_with_error.call_count == 1
    # Verify span was created for the successful retry
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 1


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_cycle_with_parent_span(
    mock_get_tracer,
    agent,
    model,
    messages,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    parent_span = MagicMock()
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span

    model.stream.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Set the parent span for this test
    agent.trace_span = parent_span

    # Call event_loop_cycle with a parent span
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    # Verify parent_span was used when creating cycle span
    mock_tracer.start_event_loop_cycle_span.assert_called_once_with(
        invocation_state=unittest.mock.ANY, parent_span=parent_span, messages=messages
    )


@pytest.mark.asyncio
async def test_request_state_initialization(alist):
    # Create a mock agent
    mock_agent = MagicMock()
    mock_agent.event_loop_metrics.start_cycle.return_value = (0, MagicMock())

    # Call without providing request_state
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=mock_agent,
        invocation_state={},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify request_state was initialized to empty dict
    assert tru_request_state == {}

    # Call with pre-existing request_state
    initial_request_state = {"key": "value"}
    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=mock_agent,
        invocation_state={"request_state": initial_request_state},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify existing request_state was preserved
    assert tru_request_state == initial_request_state


@pytest.mark.asyncio
async def test_prepare_next_cycle_in_tool_execution(agent, model, tool_stream, agenerator, alist):
    """Test that cycle ID and metrics are properly updated during tool execution."""
    model.stream.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Create a mock for recurse_event_loop to capture the invocation_state passed to it
    with unittest.mock.patch.object(strands.event_loop.event_loop, "recurse_event_loop") as mock_recurse:
        # Set up mock to return a valid response
        mock_recurse.return_value = agenerator(
            [
                (
                    "end_turn",
                    {"role": "assistant", "content": [{"text": "test text"}]},
                    strands.telemetry.metrics.EventLoopMetrics(),
                    {},
                ),
            ]
        )

        # Call event_loop_cycle which should execute a tool and then call recurse_event_loop
        stream = strands.event_loop.event_loop.event_loop_cycle(
            agent=agent,
            invocation_state={},
        )
        await alist(stream)

        assert mock_recurse.called

        # Verify required properties are present
        recursive_args = mock_recurse.call_args[1]
        assert "event_loop_parent_cycle_id" in recursive_args["invocation_state"]
        assert (
            recursive_args["invocation_state"]["event_loop_parent_cycle_id"]
            == recursive_args["invocation_state"]["event_loop_cycle_id"]
        )


@pytest.mark.asyncio
async def test_run_tool(agent, tool, alist):
    process = run_tool(
        agent,
        tool_use={"toolUseId": "tool_use_id", "name": tool.tool_name, "input": {"random_string": "a_string"}},
        invocation_state={},
    )

    tru_result = (await alist(process))[-1]
    exp_result = {"toolUseId": "tool_use_id", "status": "success", "content": [{"text": "a_string"}]}

    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_run_tool_missing_tool(agent, alist):
    process = run_tool(
        agent,
        tool_use={"toolUseId": "missing", "name": "missing", "input": {}},
        invocation_state={},
    )

    tru_events = await alist(process)
    exp_events = [
        {
            "toolUseId": "missing",
            "status": "error",
            "content": [{"text": "Unknown tool: missing"}],
        },
    ]

    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_run_tool_hooks(agent, hook_provider, tool_times_2, alist):
    """Test that the correct hooks are emitted."""

    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        invocation_state={},
    )
    await alist(process)

    assert len(hook_provider.events_received) == 2

    assert hook_provider.events_received[0] == BeforeToolInvocationEvent(
        agent=agent,
        selected_tool=tool_times_2,
        tool_use={"input": {"x": 5}, "name": "multiply_by_2", "toolUseId": "test"},
        invocation_state=ANY,
    )

    assert hook_provider.events_received[1] == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=tool_times_2,
        exception=None,
        tool_use={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        result={"toolUseId": "test", "status": "success", "content": [{"text": "10"}]},
        invocation_state=ANY,
    )


@pytest.mark.asyncio
async def test_run_tool_hooks_on_missing_tool(agent, hook_provider, alist):
    """Test that AfterToolInvocation hook is invoked even when tool throws exception."""
    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "test", "name": "missing_tool", "input": {"x": 5}},
        invocation_state={},
    )
    await alist(process)

    assert len(hook_provider.events_received) == 2

    assert hook_provider.events_received[0] == BeforeToolInvocationEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"input": {"x": 5}, "name": "missing_tool", "toolUseId": "test"},
        invocation_state=ANY,
    )

    assert hook_provider.events_received[1] == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"input": {"x": 5}, "name": "missing_tool", "toolUseId": "test"},
        invocation_state=ANY,
        result={"content": [{"text": "Unknown tool: missing_tool"}], "status": "error", "toolUseId": "test"},
        exception=None,
    )


@pytest.mark.asyncio
async def test_run_tool_hook_after_tool_invocation_on_exception(agent, tool_registry, hook_provider, alist):
    """Test that AfterToolInvocation hook is invoked even when tool throws exception."""
    error = ValueError("Tool failed")

    failing_tool = MagicMock()
    failing_tool.tool_name = "failing_tool"

    failing_tool.stream.side_effect = error

    tool_registry.register_tool(failing_tool)

    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "test", "name": "failing_tool", "input": {"x": 5}},
        invocation_state={},
    )
    await alist(process)

    assert hook_provider.events_received[1] == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=failing_tool,
        tool_use={"input": {"x": 5}, "name": "failing_tool", "toolUseId": "test"},
        invocation_state=ANY,
        result={"content": [{"text": "Error: Tool failed"}], "status": "error", "toolUseId": "test"},
        exception=error,
    )


@pytest.mark.asyncio
async def test_run_tool_hook_before_tool_invocation_updates(agent, tool_times_5, hook_registry, hook_provider, alist):
    """Test that modifying properties on BeforeToolInvocation takes effect."""

    updated_tool_use = {"toolUseId": "modified", "name": "replacement_tool", "input": {"x": 3}}

    def modify_hook(event: BeforeToolInvocationEvent):
        # Modify selected_tool to use replacement_tool
        event.selected_tool = tool_times_5
        # Modify tool_use to change toolUseId
        event.tool_use = updated_tool_use

    hook_registry.add_callback(BeforeToolInvocationEvent, modify_hook)

    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "original", "name": "original_tool", "input": {"x": 1}},
        invocation_state={},
    )
    result = (await alist(process))[-1]

    # Should use replacement_tool (5 * 3 = 15) instead of original_tool (1 * 2 = 2)
    assert result == {"toolUseId": "modified", "status": "success", "content": [{"text": "15"}]}

    assert hook_provider.events_received[1] == AfterToolInvocationEvent(
        agent=agent,
        selected_tool=tool_times_5,
        tool_use=updated_tool_use,
        invocation_state=ANY,
        result={"content": [{"text": "15"}], "status": "success", "toolUseId": "modified"},
        exception=None,
    )


@pytest.mark.asyncio
async def test_run_tool_hook_after_tool_invocation_updates(agent, tool_times_2, hook_registry, alist):
    """Test that modifying properties on AfterToolInvocation takes effect."""

    updated_result = {"toolUseId": "modified", "status": "success", "content": [{"text": "modified_result"}]}

    def modify_hook(event: AfterToolInvocationEvent):
        # Modify result to change the output
        event.result = updated_result

    hook_registry.add_callback(AfterToolInvocationEvent, modify_hook)

    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "test", "name": tool_times_2.tool_name, "input": {"x": 5}},
        invocation_state={},
    )

    result = (await alist(process))[-1]
    assert result == updated_result


@pytest.mark.asyncio
async def test_run_tool_hook_after_tool_invocation_updates_with_missing_tool(agent, hook_registry, alist):
    """Test that modifying properties on AfterToolInvocation takes effect."""

    updated_result = {"toolUseId": "modified", "status": "success", "content": [{"text": "modified_result"}]}

    def modify_hook(event: AfterToolInvocationEvent):
        # Modify result to change the output
        event.result = updated_result

    hook_registry.add_callback(AfterToolInvocationEvent, modify_hook)

    process = run_tool(
        agent=agent,
        tool_use={"toolUseId": "test", "name": "missing_tool", "input": {"x": 5}},
        invocation_state={},
    )

    result = (await alist(process))[-1]
    assert result == updated_result


@pytest.mark.asyncio
async def test_run_tool_hook_update_result_with_missing_tool(agent, tool_registry, hook_registry, alist):
    """Test that modifying properties on AfterToolInvocation takes effect."""

    @strands.tool
    def test_quota():
        return "9"

    tool_registry.register_tool(test_quota)

    class ExampleProvider(HookProvider):
        def register_hooks(self, registry: "HookRegistry") -> None:
            registry.add_callback(BeforeToolInvocationEvent, self.before_tool_call)
            registry.add_callback(AfterToolInvocationEvent, self.after_tool_call)

        def before_tool_call(self, event: BeforeToolInvocationEvent):
            if event.tool_use.get("name") == "test_quota":
                event.selected_tool = None

        def after_tool_call(self, event: AfterToolInvocationEvent):
            if event.tool_use.get("name") == "test_quota":
                event.result = {
                    "status": "error",
                    "toolUseId": "test",
                    "content": [{"text": "This tool has been used too many times!"}],
                }

    hook_registry.add_hook(ExampleProvider())

    with patch.object(strands.event_loop.event_loop, "logger") as mock_logger:
        process = run_tool(
            agent=agent,
            tool_use={"toolUseId": "test", "name": "test_quota", "input": {"x": 5}},
            invocation_state={},
        )

        result = (await alist(process))[-1]

    assert result == {
        "status": "error",
        "toolUseId": "test",
        "content": [{"text": "This tool has been used too many times!"}],
    }

    assert mock_logger.debug.call_args_list == [
        call("tool_use=<%s> | streaming", {"toolUseId": "test", "name": "test_quota", "input": {"x": 5}}),
        call(
            "tool_name=<%s>, tool_use_id=<%s> | a hook resulted in a non-existing tool call",
            "test_quota",
            "test",
        ),
    ]


@pytest.mark.asyncio
async def test_event_loop_cycle_exception_model_hooks(mock_time, agent, model, agenerator, alist, hook_provider):
    """Test that model hooks are correctly emitted even when throttled."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    exception = ModelThrottledException("ThrottlingException | ConverseStream")
    model.stream.side_effect = [
        exception,
        exception,
        exception,
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        agent=agent,
        invocation_state={},
    )
    await alist(stream)

    count, events = hook_provider.get_events()

    assert count == 8

    # 1st call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 2nd call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 3rd call - throttled
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(agent=agent, stop_response=None, exception=exception)

    # 4th call - successful
    assert next(events) == BeforeModelInvocationEvent(agent=agent)
    assert next(events) == AfterModelInvocationEvent(
        agent=agent,
        stop_response=AfterModelInvocationEvent.ModelStopResponse(
            message={"content": [{"text": "test text"}], "role": "assistant"}, stop_reason="end_turn"
        ),
        exception=None,
    )
