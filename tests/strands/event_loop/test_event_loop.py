import concurrent
import unittest.mock
from unittest.mock import MagicMock, call, patch

import pytest

import strands
import strands.telemetry
from strands.handlers.tool_handler import AgentToolHandler
from strands.tools.registry import ToolRegistry
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException, ModelThrottledException


@pytest.fixture
def mock_time():
    """Fixture to mock the time module in the error_handler."""
    with unittest.mock.patch.object(strands.event_loop.error_handler, "time") as mock:
        yield mock


@pytest.fixture
def model():
    return unittest.mock.Mock()


@pytest.fixture
def model_id():
    return "m1"


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
def callback_handler():
    return unittest.mock.Mock()


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def tool_handler(tool_registry):
    return AgentToolHandler(tool_registry)


@pytest.fixture
def tool_execution_handler():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return strands.tools.ThreadPoolExecutorWrapper(pool)


@pytest.fixture
def tool(tool_registry):
    @strands.tools.tool
    def tool_for_testing(random_string: str) -> str:
        return random_string

    function_tool = strands.tools.tools.FunctionTool(tool_for_testing)
    tool_registry.register_tool(function_tool)

    return function_tool


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


@pytest.mark.parametrize(
    ("kwargs", "exp_state"),
    [
        (
            {"request_state": {"key1": "value1"}},
            {"key1": "value1"},
        ),
        (
            {},
            {},
        ),
    ],
)
def test_initialize_state(kwargs, exp_state):
    kwargs = strands.event_loop.event_loop.initialize_state(**kwargs)

    tru_state = kwargs["request_state"]

    assert tru_state == exp_state


def test_event_loop_cycle_text_response(
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
):
    model.converse.return_value = [
        {"contentBlockDelta": {"delta": {"text": "test text"}}},
        {"contentBlockStop": {}},
    ]

    tru_stop_reason, tru_message, _, tru_request_state = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )
    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state


def test_event_loop_cycle_text_response_throttling(
    mock_time,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
):
    model.converse.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ],
    ]

    tru_stop_reason, tru_message, _, tru_request_state = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )
    exp_stop_reason = "end_turn"
    exp_message = {"role": "assistant", "content": [{"text": "test text"}]}
    exp_request_state = {}

    assert tru_stop_reason == exp_stop_reason and tru_message == exp_message and tru_request_state == exp_request_state
    # Verify that sleep was called once with the initial delay
    mock_time.sleep.assert_called_once()


def test_event_loop_cycle_exponential_backoff(
    mock_time,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
):
    """Test that the exponential backoff works correctly with multiple retries."""
    # Set up the model to raise throttling exceptions multiple times before succeeding
    model.converse.side_effect = [
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        ModelThrottledException("ThrottlingException | ConverseStream"),
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ],
    ]

    tru_stop_reason, tru_message, _, tru_request_state = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )

    # Verify the final response
    assert tru_stop_reason == "end_turn"
    assert tru_message == {"role": "assistant", "content": [{"text": "test text"}]}
    assert tru_request_state == {}

    # Verify that sleep was called with increasing delays
    # Initial delay is 4, then 8, then 16
    assert mock_time.sleep.call_count == 3
    assert mock_time.sleep.call_args_list == [call(4), call(8), call(16)]


def test_event_loop_cycle_text_response_error(
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
):
    model.converse.side_effect = RuntimeError("Unhandled error")

    with pytest.raises(RuntimeError):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            model_id=model_id,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )


def test_event_loop_cycle_tool_result(
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool_stream,
):
    model.converse.side_effect = [
        tool_stream,
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ],
    ]

    tru_stop_reason, tru_message, _, tru_request_state = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )
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


def test_event_loop_cycle_tool_result_error(
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool_stream,
):
    model.converse.side_effect = [tool_stream]

    with pytest.raises(EventLoopException):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )


def test_event_loop_cycle_tool_result_no_tool_handler(
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_execution_handler,
    tool_stream,
):
    model.converse.side_effect = [tool_stream]

    with pytest.raises(EventLoopException):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=None,
            tool_execution_handler=tool_execution_handler,
        )


def test_event_loop_cycle_tool_result_no_tool_config(
    model,
    system_prompt,
    messages,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool_stream,
):
    model.converse.side_effect = [tool_stream]

    with pytest.raises(EventLoopException):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=None,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )


def test_event_loop_cycle_stop(
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool,
):
    model.converse.side_effect = [
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
        ],
    ]

    tru_stop_reason, tru_message, _, tru_request_state = strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
        request_state={"stop_event_loop": True},
    )
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


def test_prepare_next_cycle():
    kwargs = {"event_loop_cycle_id": "c1"}
    event_loop_metrics = strands.telemetry.metrics.EventLoopMetrics()
    tru_result = strands.event_loop.event_loop.prepare_next_cycle(kwargs, event_loop_metrics)
    exp_result = {
        "event_loop_cycle_id": "c1",
        "event_loop_parent_cycle_id": "c1",
        "event_loop_metrics": event_loop_metrics,
    }

    assert tru_result == exp_result


def test_cycle_exception(
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool_stream,
):
    model.converse.side_effect = [tool_stream, tool_stream, tool_stream, ValueError("Invalid error presented")]

    with pytest.raises(EventLoopException):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            model_id=model_id,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )

    exception_calls = [
        it
        for it in callback_handler.call_args_list
        if it == call(force_stop=True, force_stop_reason="Invalid error presented")
    ]

    assert len(exception_calls) == 1


@patch("strands.event_loop.event_loop.get_tracer")
def test_event_loop_cycle_creates_spans(
    mock_get_tracer,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    mock_tracer,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    model.converse.return_value = [
        {"contentBlockDelta": {"delta": {"text": "test text"}}},
        {"contentBlockStop": {}},
    ]

    # Call event_loop_cycle
    strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )

    # Verify tracer methods were called correctly
    mock_get_tracer.assert_called_once()
    mock_tracer.start_event_loop_cycle_span.assert_called_once()
    mock_tracer.start_model_invoke_span.assert_called_once()
    mock_tracer.end_model_invoke_span.assert_called_once()
    mock_tracer.end_event_loop_cycle_span.assert_called_once()


@patch("strands.event_loop.event_loop.get_tracer")
def test_event_loop_tracing_with_model_error(
    mock_get_tracer,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    mock_tracer,
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
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            model_id=model_id,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )

    # Verify error handling span methods were called
    mock_tracer.end_span_with_error.assert_called_once_with(model_span, "Input too long", model.converse.side_effect)


@patch("strands.event_loop.event_loop.get_tracer")
def test_event_loop_tracing_with_tool_execution(
    mock_get_tracer,
    model,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    tool_stream,
    mock_tracer,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span
    model_span = MagicMock()
    mock_tracer.start_model_invoke_span.return_value = model_span

    # Set up model to return tool use and then text response
    model.converse.side_effect = [
        tool_stream,
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ],
    ]

    # Call event_loop_cycle which should execute a tool
    strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
    )

    # Verify the parent_span parameter is passed to run_tools
    # At a minimum, verify both model spans were created (one for each model invocation)
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 2


@patch("strands.event_loop.event_loop.get_tracer")
def test_event_loop_tracing_with_throttling_exception(
    mock_get_tracer,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    mock_tracer,
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
        [
            {"contentBlockDelta": {"delta": {"text": "test text"}}},
            {"contentBlockStop": {}},
        ],
    ]

    # Mock the time.sleep function to speed up the test
    with patch("strands.event_loop.error_handler.time.sleep"):
        strands.event_loop.event_loop.event_loop_cycle(
            model=model,
            model_id=model_id,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            tool_execution_handler=tool_execution_handler,
        )

    # Verify error span was created for the throttling exception
    assert mock_tracer.end_span_with_error.call_count == 1
    # Verify span was created for the successful retry
    assert mock_tracer.start_model_invoke_span.call_count == 2
    assert mock_tracer.end_model_invoke_span.call_count == 1


@patch("strands.event_loop.event_loop.get_tracer")
def test_event_loop_cycle_with_parent_span(
    mock_get_tracer,
    model,
    model_id,
    system_prompt,
    messages,
    tool_config,
    callback_handler,
    tool_handler,
    tool_execution_handler,
    mock_tracer,
):
    # Setup
    mock_get_tracer.return_value = mock_tracer
    parent_span = MagicMock()
    cycle_span = MagicMock()
    mock_tracer.start_event_loop_cycle_span.return_value = cycle_span

    model.converse.return_value = [
        {"contentBlockDelta": {"delta": {"text": "test text"}}},
        {"contentBlockStop": {}},
    ]

    # Call event_loop_cycle with a parent span
    strands.event_loop.event_loop.event_loop_cycle(
        model=model,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        tool_config=tool_config,
        callback_handler=callback_handler,
        tool_handler=tool_handler,
        tool_execution_handler=tool_execution_handler,
        event_loop_parent_span=parent_span,
    )

    # Verify parent_span was used when creating cycle span
    mock_tracer.start_event_loop_cycle_span.assert_called_once_with(
        event_loop_kwargs=unittest.mock.ANY, parent_span=parent_span, messages=messages
    )
