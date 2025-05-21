import dataclasses
import unittest

import pytest
import strands
from strands.types.streaming import Metrics, Usage


@pytest.fixture(autouse=True)
def moto_autouse(moto_env, moto_mock_aws):
    _ = moto_env
    _ = moto_mock_aws


@pytest.fixture
def trace(request):
    params = {
        "name": "t1",
        "parent_id": "p1",
        "start_time": 0,
        "raw_name": "r1",
        "metadata": {},
    }
    if hasattr(request, "param"):
        params.update(request.param)

    with unittest.mock.patch.object(strands.telemetry.metrics.uuid, "uuid4") as mock_uuid4:
        mock_uuid4.return_value = "i1"

        return strands.telemetry.metrics.Trace(**params)


@pytest.fixture
def child_trace(request):
    params = {
        "name": "c1",
        "parent_id": "p1",
        "start_time": 0,
        "raw_name": "r1",
        "metadata": {},
    }
    if hasattr(request, "param"):
        params.update(request.param)

    with unittest.mock.patch.object(strands.telemetry.metrics.uuid, "uuid4") as mock_uuid4:
        mock_uuid4.return_value = "i1"

        return strands.telemetry.metrics.Trace(**params)


@pytest.fixture
def tool(request):
    params = {
        "name": "tool1",
        "toolUseId": "123",
        "input": {},
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return params


@pytest.fixture
def tool_metrics(tool, request):
    params = {"tool": tool}
    if hasattr(request, "param"):
        params.update(request.param)

    return strands.telemetry.metrics.ToolMetrics(**params)


@pytest.fixture
def event_loop_metrics(request):
    params = {}
    if hasattr(request, "param"):
        params.update(request.param)

    return strands.telemetry.metrics.EventLoopMetrics(**params)


@pytest.fixture
def usage(request):
    params = {
        "inputTokens": 1,
        "outputTokens": 2,
        "totalTokens": 3,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return Usage(**params)


@pytest.fixture
def metrics(request):
    params = {
        "latencyMs": 1,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return Metrics(**params)


@pytest.mark.parametrize("end_time", [None, 1])
@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
def test_trace_end(mock_time, end_time, trace):
    mock_time.return_value = 1

    trace.end(end_time)

    tru_end_time = trace.end_time
    exp_end_time = 1

    assert tru_end_time == exp_end_time


def test_trace_add_child(child_trace, trace):
    trace.add_child(child_trace)

    tru_children = trace.children
    exp_children = [child_trace]

    assert tru_children == exp_children


@pytest.mark.parametrize(
    ("end_time", "exp_duration"),
    [
        (None, None),
        (1, 1),
    ],
)
def test_trace_duration(end_time, exp_duration, trace):
    if end_time is not None:
        trace.end(end_time)

    tru_duration = trace.duration()
    assert tru_duration == exp_duration


def test_trace_to_dict(trace):
    trace.end(1)

    tru_dict = trace.to_dict()
    exp_dict = {
        "id": "i1",
        "name": "t1",
        "raw_name": "r1",
        "parent_id": "p1",
        "start_time": 0,
        "end_time": 1,
        "duration": 1,
        "children": [],
        "metadata": {},
        "message": None,
    }

    assert tru_dict == exp_dict


@pytest.mark.parametrize("success", [True, False])
def test_tool_metrics_add_call(success, tool, tool_metrics):
    tool = dict(tool, **{"name": "updated"})
    duration = 1

    tool_metrics.add_call(tool, duration, success)

    tru_attrs = dataclasses.asdict(tool_metrics)
    exp_attrs = {
        "tool": tool,
        "call_count": 1,
        "success_count": success,
        "error_count": not success,
        "total_time": duration,
    }

    assert tru_attrs == exp_attrs


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
@unittest.mock.patch.object(strands.telemetry.metrics.uuid, "uuid4")
def test_event_loop_metrics_start_cycle(mock_uuid4, mock_time, event_loop_metrics):
    mock_time.return_value = 1
    mock_uuid4.return_value = "i1"

    tru_start_time, tru_cycle_trace = event_loop_metrics.start_cycle()
    exp_start_time, exp_cycle_trace = 1, strands.telemetry.metrics.Trace("Cycle 1")

    tru_attrs = {"cycle_count": event_loop_metrics.cycle_count, "traces": event_loop_metrics.traces}
    exp_attrs = {"cycle_count": 1, "traces": [tru_cycle_trace]}

    assert (
        tru_start_time == exp_start_time
        and tru_cycle_trace.to_dict() == exp_cycle_trace.to_dict()
        and tru_attrs == exp_attrs
    )


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
def test_event_loop_metrics_end_cycle(mock_time, trace, event_loop_metrics):
    mock_time.return_value = 1

    event_loop_metrics.end_cycle(start_time=0, cycle_trace=trace)

    tru_cycle_durations = event_loop_metrics.cycle_durations
    exp_cycle_durations = [1]

    assert tru_cycle_durations == exp_cycle_durations

    tru_trace_end_time = trace.end_time
    exp_trace_end_time = 1

    assert tru_trace_end_time == exp_trace_end_time


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
def test_event_loop_metrics_add_tool_usage(mock_time, trace, tool, event_loop_metrics):
    mock_time.return_value = 1

    duration = 1
    success = True
    message = {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "tool_name": "tool1"}}]}

    event_loop_metrics.add_tool_usage(tool, duration, trace, success, message)

    tru_event_loop_metrics_attrs = {"tool_metrics": event_loop_metrics.tool_metrics}
    exp_event_loop_metrics_attrs = {
        "tool_metrics": {
            "tool1": strands.telemetry.metrics.ToolMetrics(
                tool=tool,
                call_count=1,
                success_count=1,
                error_count=0,
                total_time=duration,
            ),
        }
    }

    assert tru_event_loop_metrics_attrs == exp_event_loop_metrics_attrs

    tru_trace_attrs = {
        "metadata": trace.metadata,
        "raw_name": trace.raw_name,
        "end_time": trace.end_time,
    }
    exp_trace_attrs = {
        "metadata": {
            "toolUseId": "123",
            "tool_name": "tool1",
        },
        "raw_name": "tool1 - 123",
        "end_time": 1,
    }

    assert tru_trace_attrs == exp_trace_attrs


def test_event_loop_metrics_update_usage(usage, event_loop_metrics):
    for _ in range(3):
        event_loop_metrics.update_usage(usage)

    tru_usage = event_loop_metrics.accumulated_usage
    exp_usage = Usage(
        inputTokens=3,
        outputTokens=6,
        totalTokens=9,
    )

    assert tru_usage == exp_usage


def test_event_loop_metrics_update_metrics(metrics, event_loop_metrics):
    for _ in range(3):
        event_loop_metrics.update_metrics(metrics)

    tru_metrics = event_loop_metrics.accumulated_metrics
    exp_metrics = Metrics(
        latencyMs=3,
    )

    assert tru_metrics == exp_metrics


def test_event_loop_metrics_get_summary(trace, tool, event_loop_metrics):
    duration = 1
    success = True
    message = {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "tool_name": "tool1"}}]}

    event_loop_metrics.add_tool_usage(tool, duration, trace, success, message)

    tru_summary = event_loop_metrics.get_summary()
    exp_summary = {
        "accumulated_metrics": {
            "latencyMs": 0,
        },
        "accumulated_usage": {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        },
        "average_cycle_time": 0,
        "tool_usage": {
            "tool1": {
                "execution_stats": {
                    "average_time": 1,
                    "call_count": 1,
                    "error_count": 0,
                    "success_count": 1,
                    "success_rate": 1,
                    "total_time": 1,
                },
                "tool_info": {
                    "input_params": {},
                    "name": "tool1",
                    "tool_use_id": "123",
                },
            },
        },
        "total_cycles": 0,
        "total_duration": 0,
        "traces": [],
    }

    assert tru_summary == exp_summary


@pytest.mark.parametrize(
    ("trace", "child_trace", "tool_metrics", "exp_str"),
    [
        (
            {},
            {},
            {},
            "Event Loop Metrics Summary:\n"
            "├─ Cycles: total=0, avg_time=0.000s, total_time=0.000s\n"
            "├─ Tokens: in=0, out=0, total=0\n"
            "├─ Bedrock Latency: 0ms\n"
            "├─ Tool Usage:\n"
            "   └─ tool1:\n"
            "      ├─ Stats: calls=0, success=0\n"
            "      │         errors=0, success_rate=0.0%\n"
            "      ├─ Timing: avg=0.000s, total=0.000s\n"
            "      └─ Tool Calls:\n"
            "├─ Execution Trace:\n"
            "   └─ r1 - Duration: None\n"
            "      └─ r1 - Duration: None",
        ),
        (
            {"raw_name": "t1 - tooluse_"},
            {"metadata": {"tool_name": "tool1", "toolUseId": "123"}},
            {},
            "Event Loop Metrics Summary:\n"
            "├─ Cycles: total=0, avg_time=0.000s, total_time=0.000s\n"
            "├─ Tokens: in=0, out=0, total=0\n"
            "├─ Bedrock Latency: 0ms\n"
            "├─ Tool Usage:\n"
            "   └─ tool1:\n"
            "      ├─ Stats: calls=0, success=0\n"
            "      │         errors=0, success_rate=0.0%\n"
            "      ├─ Timing: avg=0.000s, total=0.000s\n"
            "      └─ Tool Calls:\n"
            "         ├─ 123: tool1\n"
            "├─ Execution Trace:\n"
            "   └─ t1 - tooluse_ - Duration: None\n"
            "      └─ r1 - 123 - Duration: None",
        ),
    ],
    indirect=["trace", "child_trace", "tool_metrics"],
)
def test_metrics_to_string(trace, child_trace, tool_metrics, exp_str, event_loop_metrics):
    trace.add_child(child_trace)

    event_loop_metrics.traces = [trace]
    event_loop_metrics.tool_metrics = {tool_metrics.tool["name"]: tool_metrics}

    tru_str = strands.telemetry.metrics.metrics_to_string(event_loop_metrics)

    assert tru_str == exp_str
