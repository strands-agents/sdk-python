import concurrent
import functools
import unittest.mock

import pytest

import strands
import strands.telemetry
from strands.types.content import Message


@pytest.fixture
def tool_handler(request):
    def handler(tool_use):
        return {
            **params,
            "toolUseId": tool_use["toolUseId"],
        }

    params = {
        "content": [{"text": "test result"}],
        "status": "success",
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return handler


@pytest.fixture
def tool_use():
    return {"toolUseId": "t1", "name": "test_tool", "input": {"key": "value"}}


@pytest.fixture
def tool_uses(request, tool_use):
    return request.param if hasattr(request, "param") else [tool_use]


@pytest.fixture
def event_loop_metrics():
    return strands.telemetry.metrics.EventLoopMetrics()


@pytest.fixture
def request_state():
    return {}


@pytest.fixture
def invalid_tool_use_ids(request):
    return request.param if hasattr(request, "param") else []


@unittest.mock.patch.object(strands.telemetry.metrics, "uuid4", return_value="trace1")
@pytest.fixture
def cycle_trace():
    return strands.telemetry.metrics.Trace(name="test trace", raw_name="raw_name")


@pytest.fixture
def parallel_tool_executor(request):
    params = {
        "max_workers": 1,
        "timeout": None,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    as_completed = functools.partial(concurrent.futures.as_completed, timeout=params["timeout"])

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=params["max_workers"])
    wrapper = strands.tools.ThreadPoolExecutorWrapper(pool)

    with unittest.mock.patch.object(wrapper, "as_completed", side_effect=as_completed):
        yield wrapper


def test_run_tools(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    tool_results = []

    failed = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parallel_tool_executor,
    )
    assert not failed

    tru_results = tool_results
    exp_results = [
        {
            "content": [
                {
                    "text": "test result",
                },
            ],
            "status": "success",
            "toolUseId": "t1",
        },
    ]

    assert tru_results == exp_results


@pytest.mark.parametrize("invalid_tool_use_ids", [["t1"]], indirect=True)
def test_run_tools_invalid_tool(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    tool_results = []

    failed = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parallel_tool_executor,
    )
    assert failed

    tru_results = tool_results
    exp_results = []

    assert tru_results == exp_results


@pytest.mark.parametrize("tool_handler", [{"status": "failed"}], indirect=True)
def test_run_tools_failed_tool(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    tool_results = []

    failed = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parallel_tool_executor,
    )
    assert failed

    tru_results = tool_results
    exp_results = [
        {
            "content": [
                {
                    "text": "test result",
                },
            ],
            "status": "failed",
            "toolUseId": "t1",
        },
    ]

    assert tru_results == exp_results


@pytest.mark.parametrize(
    ("tool_uses", "invalid_tool_use_ids"),
    [
        (
            [
                {
                    "toolUseId": "t1",
                    "name": "test_tool_success",
                    "input": {"key": "value1"},
                },
                {
                    "toolUseId": "t2",
                    "name": "test_tool_invalid",
                    "input": {"key": "value2"},
                },
            ],
            ["t2"],
        ),
    ],
    indirect=True,
)
def test_run_tools_sequential(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
):
    tool_results = []

    failed = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        None,  # parallel_tool_executor
    )
    assert failed

    tru_results = tool_results
    exp_results = [
        {
            "content": [
                {
                    "text": "test result",
                },
            ],
            "status": "success",
            "toolUseId": "t1",
        },
    ]

    assert tru_results == exp_results


def test_validate_and_prepare_tools():
    message: Message = {
        "role": "assistant",
        "content": [
            {"text": "value"},
            {"toolUse": {"toolUseId": "t1", "name": "test_tool", "input": {"key": "value"}}},
            {"toolUse": {"toolUseId": "t2-invalid"}},
        ],
    }

    tool_uses = []
    tool_results = []
    invalid_tool_use_ids = []

    strands.tools.executor.validate_and_prepare_tools(message, tool_uses, tool_results, invalid_tool_use_ids)

    tru_tool_uses, tru_tool_results, tru_invalid_tool_use_ids = tool_uses, tool_results, invalid_tool_use_ids
    exp_tool_uses = [
        {
            "input": {
                "key": "value",
            },
            "name": "test_tool",
            "toolUseId": "t1",
        },
        {
            "name": "INVALID_TOOL_NAME",
            "toolUseId": "t2-invalid",
        },
    ]
    exp_tool_results = [
        {
            "content": [
                {
                    "text": "Error: tool name missing",
                },
            ],
            "status": "error",
            "toolUseId": "t2-invalid",
        },
    ]
    exp_invalid_tool_use_ids = ["t2-invalid"]

    assert tru_tool_uses == exp_tool_uses
    assert tru_tool_results == exp_tool_results
    assert tru_invalid_tool_use_ids == exp_invalid_tool_use_ids


@unittest.mock.patch("strands.tools.executor.get_tracer")
def test_run_tools_creates_and_ends_span_on_success(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    """Test that run_tools creates and ends a span on successful execution."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_tool_call_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock parent span
    parent_span = unittest.mock.MagicMock()

    tool_results = []

    # Run the tool
    strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
        parallel_tool_executor,
    )

    # Verify span was created with the parent span
    mock_tracer.start_tool_call_span.assert_called_once_with(tool_uses[0], parent_span)

    # Verify span was ended with the tool result
    mock_tracer.end_tool_call_span.assert_called_once()
    args, _ = mock_tracer.end_tool_call_span.call_args
    assert args[0] == mock_span
    assert args[1]["status"] == "success"
    assert args[1]["content"][0]["text"] == "test result"


@unittest.mock.patch("strands.tools.executor.get_tracer")
@pytest.mark.parametrize("tool_handler", [{"status": "failed"}], indirect=True)
def test_run_tools_creates_and_ends_span_on_failure(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    """Test that run_tools creates and ends a span on tool failure."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_tool_call_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Setup mock parent span
    parent_span = unittest.mock.MagicMock()

    tool_results = []

    # Run the tool
    strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
        parallel_tool_executor,
    )

    # Verify span was created with the parent span
    mock_tracer.start_tool_call_span.assert_called_once_with(tool_uses[0], parent_span)

    # Verify span was ended with the tool result
    mock_tracer.end_tool_call_span.assert_called_once()
    args, _ = mock_tracer.end_tool_call_span.call_args
    assert args[0] == mock_span
    assert args[1]["status"] == "failed"


@unittest.mock.patch("strands.tools.executor.get_tracer")
def test_run_tools_handles_exception_in_tool_execution(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    """Test that run_tools properly handles exceptions during tool execution."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_tool_call_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Make the tool handler throw an exception
    exception = ValueError("Test tool execution error")
    mock_handler = unittest.mock.MagicMock(side_effect=exception)

    tool_results = []

    # Run the tool - the exception should be caught inside run_tools and not propagate
    # because of the try-except block in the new implementation
    failed = strands.tools.executor.run_tools(
        mock_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        None,
        parallel_tool_executor,
    )

    # Tool execution should have failed
    assert failed

    # Verify span was created
    mock_tracer.start_tool_call_span.assert_called_once()

    # Verify span was ended with the error
    mock_tracer.end_span_with_error.assert_called_once_with(mock_span, str(exception), exception)


@unittest.mock.patch("strands.tools.executor.get_tracer")
def test_run_tools_with_invalid_tool_use_id_still_creates_span(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    cycle_trace,
    parallel_tool_executor,
):
    """Test that run_tools creates a span even when the tool use ID is invalid."""
    # Setup mock tracer and span
    mock_tracer = unittest.mock.MagicMock()
    mock_span = unittest.mock.MagicMock()
    mock_tracer.start_tool_call_span.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Mark the tool use ID as invalid
    invalid_tool_use_ids = [tool_uses[0]["toolUseId"]]

    tool_results = []

    # Run the tool
    strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        None,
        parallel_tool_executor,
    )

    # Verify span was created
    mock_tracer.start_tool_call_span.assert_called_once_with(tool_uses[0], None)

    # Verify span was ended even though the tool wasn't executed
    mock_tracer.end_tool_call_span.assert_called_once()


@unittest.mock.patch("strands.tools.executor.get_tracer")
@pytest.mark.parametrize(
    ("tool_uses", "invalid_tool_use_ids"),
    [
        (
            [
                {
                    "toolUseId": "t1",
                    "name": "test_tool_success",
                    "input": {"key": "value1"},
                },
                {
                    "toolUseId": "t2",
                    "name": "test_tool_also_success",
                    "input": {"key": "value2"},
                },
            ],
            [],
        ),
    ],
    indirect=True,
)
def test_run_tools_parallel_execution_with_spans(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    request_state,
    invalid_tool_use_ids,
    cycle_trace,
    parallel_tool_executor,
):
    """Test that spans are created and ended for each tool in parallel execution."""
    # Setup mock tracer and spans
    mock_tracer = unittest.mock.MagicMock()
    mock_span1 = unittest.mock.MagicMock()
    mock_span2 = unittest.mock.MagicMock()
    mock_tracer.start_tool_call_span.side_effect = [mock_span1, mock_span2]
    mock_get_tracer.return_value = mock_tracer

    # Setup mock parent span
    parent_span = unittest.mock.MagicMock()

    tool_results = []

    # Run the tools
    strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        request_state,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
        parallel_tool_executor,
    )

    # Verify spans were created for both tools
    assert mock_tracer.start_tool_call_span.call_count == 2
    mock_tracer.start_tool_call_span.assert_has_calls(
        [
            unittest.mock.call(tool_uses[0], parent_span),
            unittest.mock.call(tool_uses[1], parent_span),
        ],
        any_order=True,
    )

    # Verify spans were ended for both tools
    assert mock_tracer.end_tool_call_span.call_count == 2
