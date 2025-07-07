import concurrent
import unittest.mock
from unittest.mock import MagicMock, call, patch

import pytest

import strands
import strands.telemetry
from strands.handlers.tool_handler import AgentToolHandler
from strands.telemetry.metrics import EventLoopMetrics
from strands.tools.registry import ToolRegistry
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException, ModelThrottledException


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
def tool_config():
    return {"tools": [{"toolSpec": {"name": "tool_for_testing"}}], "toolChoice": {"auto": {}}}


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def tool_handler(tool_registry):
    return AgentToolHandler(tool_registry)


@pytest.fixture
def thread_pool():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def tool(tool_registry):
    @strands.tools.tool
    def tool_for_testing(random_string: str) -> str:
        return random_string

    tool_registry.register_tool(tool_for_testing)

    return tool_for_testing


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
def agent():
    mock = unittest.mock.Mock()
    mock.config.cache_points = []

    return mock


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer.start_event_loop_cycle_span.return_value = MagicMock()
    tracer.start_model_invoke_span.return_value = MagicMock()
    return tracer


@pytest.mark.asyncio
async def test_event_loop_cycle_text_response(
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    agenerator,
    alist,
):
    model.converse.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    agenerator,
    alist,
):
    model.converse.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    agenerator,
    alist,
):
    """Test that the exponential backoff works correctly with multiple retries."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    model.converse.side_effect = [
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
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    alist,
):
    model.converse.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
    ]

    with pytest.raises(ModelThrottledException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    alist,
):
    model.converse.side_effect = RuntimeError("Unhandled error")

    with pytest.raises(RuntimeError):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result(
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    tool_stream,
    agenerator,
    alist,
):
    model.converse.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockDelta": {"delta": {"text": "test text"}}},
                {"contentBlockStop": {}},
            ]
        ),
    ]

    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
    )
    events = await alist(stream)
    tru_stop_reason, tru_message, _, tru_request_state = events[-1]["stop"]

    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state

    model.converse.assert_called_with(
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
        [{"name": "tool_for_testing"}],
        "p1",
    )


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_error(
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    tool_stream,
    agenerator,
    alist,
):
    model.converse.side_effect = [agenerator(tool_stream)]

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_no_tool_handler(
    model,
    system_prompt,
    messages,
    tool_config,
    thread_pool,
    tool_stream,
    agenerator,
    alist,
):
    model.converse.side_effect = [agenerator(tool_stream)]

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=None,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_tool_result_no_tool_config(
    model,
    system_prompt,
    messages,
    tool_handler,
    thread_pool,
    tool_stream,
    agenerator,
    alist,
):
    model.converse.side_effect = [agenerator(tool_stream)]

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=None,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)


@pytest.mark.asyncio
async def test_event_loop_cycle_stop(
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    tool,
    agenerator,
    alist,
):
    model.converse.side_effect = [
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
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={"request_state": {"stop_event_loop": True}},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    tool_stream,
    agenerator,
):
    model.converse.side_effect = [
        agenerator(tool_stream),
        agenerator(tool_stream),
        agenerator(tool_stream),
        ValueError("Invalid error presented"),
    ]

    tru_stop_event = None
    exp_stop_event = {"callback": {"force_stop": True, "force_stop_reason": "Invalid error presented"}}

    with pytest.raises(EventLoopException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        async for event in stream:
            tru_stop_event = event

    assert tru_stop_event == exp_stop_event


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_cycle_creates_spans(
    mock_get_tracer,
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
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

    model.converse.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Call event_loop_cycle
    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
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
    model.converse.side_effect = ContextWindowOverflowException("Input too long")

    # Call event_loop_cycle, expecting it to handle the exception
    with pytest.raises(ContextWindowOverflowException):
        stream = strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)

    # Verify error handling span methods were called
    mock_tracer.end_span_with_error.assert_called_once_with(model_span, "Input too long", model.converse.side_effect)


@patch("strands.event_loop.event_loop.get_tracer")
@pytest.mark.asyncio
async def test_event_loop_tracing_with_tool_execution(
    mock_get_tracer,
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
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
    model.converse.side_effect = [
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
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
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
    model.converse.side_effect = [
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
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            tool_handler=tool_handler,
            thread_pool=thread_pool,
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
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
    model,
    system_prompt,
    messages,
    tool_config,
    tool_handler,
    thread_pool,
    mock_tracer,
    agenerator,
    alist,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    parent_span = MagicMock()
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span

    model.converse.return_value = agenerator(
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ]
    )

    # Call event_loop_cycle with a parent span
    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        tool_handler=tool_handler,
        thread_pool=thread_pool,
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=parent_span,
        kwargs={},
    )
    await alist(stream)

    # Verify parent_span was used when creating cycle span
    mock_tracer.start_event_loop_cycle_span.assert_called_once_with(
        event_loop_kwargs=unittest.mock.ANY, parent_span=parent_span, messages=messages
    )


@pytest.mark.asyncio
async def test_request_state_initialization(alist):
    # Call without providing request_state
    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=MagicMock(),
        system_prompt=MagicMock(),
        messages=MagicMock(),
        tool_config=MagicMock(),
        tool_handler=MagicMock(),
        thread_pool=MagicMock(),
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify request_state was initialized to empty dict
    assert tru_request_state == {}

    # Call with pre-existing request_state
    initial_request_state = {"key": "value"}
    stream = strands.event_loop.event_loop.event_loop_cycle(
        model=MagicMock(),
        system_prompt=MagicMock(),
        messages=MagicMock(),
        tool_config=MagicMock(),
        tool_handler=MagicMock(),
        thread_pool=MagicMock(),
        event_loop_metrics=EventLoopMetrics(),
        event_loop_parent_span=None,
        kwargs={"request_state": initial_request_state},
    )
    events = await alist(stream)
    _, _, _, tru_request_state = events[-1]["stop"]

    # Verify existing request_state was preserved
    assert tru_request_state == initial_request_state


@pytest.mark.asyncio
async def test_prepare_next_cycle_in_tool_execution(model, tool_stream, agenerator, alist):
    """Test that cycle ID and metrics are properly updated during tool execution."""
    model.converse.side_effect = [
        agenerator(tool_stream),
        agenerator(
            [
                {"contentBlockStop": {}},
            ]
        ),
    ]

    # Create a mock for recurse_event_loop to capture the kwargs passed to it
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
            model=model,
            system_prompt=MagicMock(),
            messages=MagicMock(),
            tool_config=MagicMock(),
            tool_handler=MagicMock(),
            thread_pool=MagicMock(),
            event_loop_metrics=EventLoopMetrics(),
            event_loop_parent_span=None,
            kwargs={},
        )
        await alist(stream)

        assert mock_recurse.called

        # Verify required properties are present
        recursive_args = mock_recurse.call_args[1]
        assert "event_loop_parent_cycle_id" in recursive_args["kwargs"]
        assert recursive_args["kwargs"]["event_loop_parent_cycle_id"] == recursive_args["kwargs"]["event_loop_cycle_id"]
