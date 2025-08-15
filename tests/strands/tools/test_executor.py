import unittest.mock
import uuid

import pytest

import strands
import strands.telemetry
from strands.types.content import Message


@pytest.fixture(autouse=True)
def moto_autouse(moto_env):
    _ = moto_env


@pytest.fixture
def tool_handler(request):
    async def handler(tool_use):
        yield {"event": "abc"}
        yield {
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
def mock_metrics_client():
    with unittest.mock.patch("strands.telemetry.MetricsClient") as mock_metrics_client:
        yield mock_metrics_client


@pytest.fixture
def event_loop_metrics():
    return strands.telemetry.metrics.EventLoopMetrics()


@pytest.fixture
def invalid_tool_use_ids(request):
    return request.param if hasattr(request, "param") else []


@pytest.fixture
def cycle_trace():
    with unittest.mock.patch.object(uuid, "uuid4", return_value="trace1"):
        return strands.telemetry.metrics.Trace(name="test trace", raw_name="raw_name")


@pytest.mark.asyncio
async def test_run_tools(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
):
    tool_results = []

    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
    )

    tru_events = await alist(stream)
    exp_events = [
        {"event": "abc"},
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

    tru_results = tool_results
    exp_results = [exp_events[-1]]

    assert tru_events == exp_events and tru_results == exp_results


@pytest.mark.parametrize("invalid_tool_use_ids", [["t1"]], indirect=True)
@pytest.mark.asyncio
async def test_run_tools_invalid_tool(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
):
    tool_results = []

    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
    )
    await alist(stream)

    tru_results = tool_results
    exp_results = []

    assert tru_results == exp_results


@pytest.mark.parametrize("tool_handler", [{"status": "failed"}], indirect=True)
@pytest.mark.asyncio
async def test_run_tools_failed_tool(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
):
    tool_results = []

    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
    )
    await alist(stream)

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
@pytest.mark.asyncio
async def test_run_tools_sequential(
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
):
    tool_results = []

    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        None,  # tool_pool
    )
    await alist(stream)

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
@pytest.mark.asyncio
async def test_run_tools_creates_and_ends_span_on_success(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    mock_metrics_client,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
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
    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
    )
    await alist(stream)

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
@pytest.mark.asyncio
async def test_run_tools_creates_and_ends_span_on_failure(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
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
    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
    )
    await alist(stream)

    # Verify span was created with the parent span
    mock_tracer.start_tool_call_span.assert_called_once_with(tool_uses[0], parent_span)

    # Verify span was ended with the tool result
    mock_tracer.end_tool_call_span.assert_called_once()
    args, _ = mock_tracer.end_tool_call_span.call_args
    assert args[0] == mock_span
    assert args[1]["status"] == "failed"


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
@pytest.mark.asyncio
async def test_run_tools_concurrent_execution_with_spans(
    mock_get_tracer,
    tool_handler,
    tool_uses,
    event_loop_metrics,
    invalid_tool_use_ids,
    cycle_trace,
    alist,
):
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
    stream = strands.tools.executor.run_tools(
        tool_handler,
        tool_uses,
        event_loop_metrics,
        invalid_tool_use_ids,
        tool_results,
        cycle_trace,
        parent_span,
    )
    await alist(stream)

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
