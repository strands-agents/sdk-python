import dataclasses
import unittest
from unittest import mock

import pytest
from opentelemetry.metrics._internal import _ProxyMeter
from opentelemetry.sdk.metrics import MeterProvider

import strands
from strands.telemetry import MetricsClient
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
        "cacheWriteInputTokens": 2,
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


@pytest.fixture
def metrics_with_ttfb(request):
    params = {
        "latencyMs": 1,
        "timeToFirstByteMs": 10,
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


@pytest.fixture
def mock_get_meter_provider():
    with mock.patch("strands.telemetry.metrics.metrics_api.get_meter_provider") as mock_get_meter_provider:
        MetricsClient._instance = None
        meter_provider_mock = mock.MagicMock(spec=MeterProvider)

        mock_meter = mock.MagicMock()
        mock_create_counter = mock.MagicMock()
        mock_meter.create_counter.return_value = mock_create_counter

        # Create separate mock objects for each histogram call
        mock_meter.create_histogram.side_effect = lambda *args, **kwargs: mock.MagicMock()
        meter_provider_mock.get_meter.return_value = mock_meter

        mock_get_meter_provider.return_value = meter_provider_mock

        yield mock_get_meter_provider


@pytest.fixture
def mock_sdk_meter_provider():
    with mock.patch("strands.telemetry.metrics.metrics_sdk.MeterProvider") as mock_meter_provider:
        yield mock_meter_provider


@pytest.fixture
def mock_resource():
    with mock.patch("opentelemetry.sdk.resources.Resource") as mock_resource:
        yield mock_resource


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
def test_tool_metrics_add_call(success, tool, tool_metrics, mock_get_meter_provider):
    tool = dict(tool, **{"name": "updated"})
    duration = 1
    metrics_client = MetricsClient()

    attributes = {"foo": "bar"}

    tool_metrics.add_call(tool, duration, success, metrics_client, attributes=attributes)

    tru_attrs = dataclasses.asdict(tool_metrics)
    exp_attrs = {
        "tool": tool,
        "call_count": 1,
        "success_count": success,
        "error_count": not success,
        "total_time": duration,
    }

    mock_get_meter_provider.return_value.get_meter.assert_called()
    metrics_client.tool_call_count.add.assert_called_with(1, attributes=attributes)
    metrics_client.tool_duration.record.assert_called_with(duration, attributes=attributes)
    if success:
        metrics_client.tool_success_count.add.assert_called_with(1, attributes=attributes)
    assert tru_attrs == exp_attrs


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
@unittest.mock.patch.object(strands.telemetry.metrics.uuid, "uuid4")
def test_event_loop_metrics_start_cycle(mock_uuid4, mock_time, event_loop_metrics, mock_get_meter_provider):
    mock_time.return_value = 1
    mock_event_loop_cycle_id = "i1"
    mock_uuid4.return_value = mock_event_loop_cycle_id

    # Reset must be called first
    event_loop_metrics.reset_usage_metrics()

    tru_start_time, tru_cycle_trace = event_loop_metrics.start_cycle(
        attributes={"event_loop_cycle_id": mock_event_loop_cycle_id}
    )
    exp_start_time, exp_cycle_trace = 1, strands.telemetry.metrics.Trace("Cycle 1")

    tru_attrs = {"cycle_count": event_loop_metrics.cycle_count, "traces": event_loop_metrics.traces}
    exp_attrs = {"cycle_count": 1, "traces": [tru_cycle_trace]}

    mock_get_meter_provider.return_value.get_meter.assert_called()
    event_loop_metrics._metrics_client.event_loop_cycle_count.add.assert_called()
    assert (
        tru_start_time == exp_start_time
        and tru_cycle_trace.to_dict() == exp_cycle_trace.to_dict()
        and tru_attrs == exp_attrs
    )

    assert len(event_loop_metrics.agent_invocations) == 1
    assert len(event_loop_metrics.agent_invocations[0].cycles) == 1
    assert event_loop_metrics.agent_invocations[0].cycles[0].event_loop_cycle_id == "i1"
    assert event_loop_metrics.agent_invocations[0].cycles[0].usage["inputTokens"] == 0
    assert event_loop_metrics.agent_invocations[0].cycles[0].usage["outputTokens"] == 0
    assert event_loop_metrics.agent_invocations[0].cycles[0].usage["totalTokens"] == 0


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
def test_event_loop_metrics_end_cycle(mock_time, trace, event_loop_metrics, mock_get_meter_provider):
    mock_time.return_value = 1

    attributes = {"foo": "bar"}
    event_loop_metrics.end_cycle(start_time=0, cycle_trace=trace, attributes=attributes)

    tru_cycle_durations = event_loop_metrics.cycle_durations
    exp_cycle_durations = [1]

    assert tru_cycle_durations == exp_cycle_durations

    tru_trace_end_time = trace.end_time
    exp_trace_end_time = 1

    assert tru_trace_end_time == exp_trace_end_time

    mock_get_meter_provider.return_value.get_meter.assert_called()
    metrics_client = event_loop_metrics._metrics_client
    metrics_client.event_loop_end_cycle.add.assert_called_with(1, attributes)
    metrics_client.event_loop_cycle_duration.record.assert_called()


@unittest.mock.patch.object(strands.telemetry.metrics.time, "time")
def test_event_loop_metrics_add_tool_usage(mock_time, trace, tool, event_loop_metrics, mock_get_meter_provider):
    mock_time.return_value = 1
    duration = 1
    success = True
    message = {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "tool_name": "tool1"}}]}

    event_loop_metrics.add_tool_usage(tool, duration, trace, success, message)

    mock_get_meter_provider.return_value.get_meter.assert_called()

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


def test_event_loop_metrics_update_usage(usage, event_loop_metrics, mock_get_meter_provider):
    event_loop_metrics.reset_usage_metrics()
    event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "test-cycle"})

    for _ in range(3):
        event_loop_metrics.update_usage(usage)

    tru_usage = event_loop_metrics.accumulated_usage
    exp_usage = Usage(inputTokens=3, outputTokens=6, totalTokens=9, cacheWriteInputTokens=6)

    assert tru_usage == exp_usage

    assert event_loop_metrics.latest_agent_invocation.usage == exp_usage

    assert len(event_loop_metrics.agent_invocations) == 1
    assert len(event_loop_metrics.agent_invocations[0].cycles) == 1
    assert event_loop_metrics.agent_invocations[0].cycles[0].event_loop_cycle_id == "test-cycle"
    assert event_loop_metrics.agent_invocations[0].cycles[0].usage == exp_usage

    mock_get_meter_provider.return_value.get_meter.assert_called()
    metrics_client = event_loop_metrics._metrics_client
    metrics_client.event_loop_input_tokens.record.assert_called()
    metrics_client.event_loop_output_tokens.record.assert_called()
    metrics_client.event_loop_cache_write_input_tokens.record.assert_called()


def test_event_loop_metrics_update_metrics(metrics_with_ttfb, event_loop_metrics, mock_get_meter_provider):
    for _ in range(3):
        event_loop_metrics.update_metrics(metrics_with_ttfb)

    tru_metrics = event_loop_metrics.accumulated_metrics
    exp_metrics = Metrics(
        latencyMs=3,
    )

    assert tru_metrics == exp_metrics
    mock_get_meter_provider.return_value.get_meter.assert_called()
    event_loop_metrics._metrics_client.event_loop_latency.record.assert_called_with(1)
    event_loop_metrics._metrics_client.model_time_to_first_token.record.assert_called_with(10)


def test_event_loop_metrics_get_summary(trace, tool, event_loop_metrics, mock_get_meter_provider):
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
        "agent_invocations": [],
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
        "initial_input_tokens": 0,
        "final_input_tokens": 0,
        "context_growth_tokens": 0,
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


def test_setup_meter_if_meter_provider_is_set(
    mock_get_meter_provider,
    mock_resource,
):
    """Test global meter_provider and meter are used"""
    mock_resource_instance = mock.MagicMock()
    mock_resource.create.return_value = mock_resource_instance

    metrics_client = MetricsClient()

    mock_get_meter_provider.assert_called()
    mock_get_meter_provider.return_value.get_meter.assert_called()

    assert metrics_client is not None


def test_use_ProxyMeter_if_no_global_meter_provider():
    """Return _ProxyMeter"""
    # Reset the singleton instance
    strands.telemetry.metrics.MetricsClient._instance = None

    # Create a new instance which should use the real _ProxyMeter
    metrics_client = MetricsClient()

    # Verify it's using a _ProxyMeter
    assert isinstance(metrics_client.meter, _ProxyMeter)


def test_latest_agent_invocation_property(usage, event_loop_metrics, mock_get_meter_provider):
    """Test the latest_agent_invocation property getter"""
    # Initially, no invocations exist
    assert event_loop_metrics.latest_agent_invocation is None

    event_loop_metrics.reset_usage_metrics()
    event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-1"})
    event_loop_metrics.update_usage(usage)

    # latest_agent_invocation should return the first invocation
    current = event_loop_metrics.latest_agent_invocation
    assert current is not None
    assert current.usage["inputTokens"] == 1
    assert len(current.cycles) == 1

    event_loop_metrics.reset_usage_metrics()
    event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-2"})
    usage2 = Usage(inputTokens=10, outputTokens=20, totalTokens=30)
    event_loop_metrics.update_usage(usage2)

    # Should return the second invocation
    current = event_loop_metrics.latest_agent_invocation
    assert current is not None
    assert current.usage["inputTokens"] == 10
    assert len(current.cycles) == 1

    assert len(event_loop_metrics.agent_invocations) == 2

    assert current is event_loop_metrics.agent_invocations[-1]


def test_reset_usage_metrics(usage, event_loop_metrics, mock_get_meter_provider):
    """Test that reset_usage_metrics creates a new agent invocation but preserves accumulated_usage"""
    # Add some usage across multiple cycles in first invocation
    event_loop_metrics.reset_usage_metrics()
    event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-1"})
    event_loop_metrics.update_usage(usage)

    event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-2"})
    usage2 = Usage(inputTokens=10, outputTokens=20, totalTokens=30)
    event_loop_metrics.update_usage(usage2)

    assert len(event_loop_metrics.agent_invocations) == 1
    assert event_loop_metrics.latest_agent_invocation.usage["inputTokens"] == 11
    assert len(event_loop_metrics.latest_agent_invocation.cycles) == 2
    assert event_loop_metrics.accumulated_usage["inputTokens"] == 11

    # Reset - creates a new invocation
    event_loop_metrics.reset_usage_metrics()

    assert len(event_loop_metrics.agent_invocations) == 2

    assert event_loop_metrics.latest_agent_invocation.usage["inputTokens"] == 0
    assert event_loop_metrics.latest_agent_invocation.usage["outputTokens"] == 0
    assert event_loop_metrics.latest_agent_invocation.usage["totalTokens"] == 0
    assert len(event_loop_metrics.latest_agent_invocation.cycles) == 0

    # Verify first invocation data is preserved
    assert event_loop_metrics.agent_invocations[0].usage["inputTokens"] == 11
    assert len(event_loop_metrics.agent_invocations[0].cycles) == 2

    # Verify accumulated_usage is NOT cleared
    assert event_loop_metrics.accumulated_usage["inputTokens"] == 11


class TestAgentInvocationContextSizeMetrics:
    """Tests for AgentInvocation context size metrics."""

    def test_initial_input_tokens_no_cycles(self):
        """Test initial_input_tokens returns 0 when no cycles exist."""
        invocation = strands.telemetry.metrics.AgentInvocation()
        assert invocation.initial_input_tokens == 0

    def test_final_input_tokens_no_cycles(self):
        """Test final_input_tokens returns 0 when no cycles exist."""
        invocation = strands.telemetry.metrics.AgentInvocation()
        assert invocation.final_input_tokens == 0

    def test_context_growth_tokens_no_cycles(self):
        """Test context_growth_tokens returns 0 when no cycles exist."""
        invocation = strands.telemetry.metrics.AgentInvocation()
        assert invocation.context_growth_tokens == 0

    def test_initial_input_tokens_single_cycle(self):
        """Test initial_input_tokens with a single cycle."""
        cycle = strands.telemetry.metrics.EventLoopCycleMetric(
            event_loop_cycle_id="cycle-1",
            usage=Usage(inputTokens=1000, outputTokens=100, totalTokens=1100),
        )
        invocation = strands.telemetry.metrics.AgentInvocation(cycles=[cycle])
        assert invocation.initial_input_tokens == 1000

    def test_final_input_tokens_single_cycle(self):
        """Test final_input_tokens with a single cycle (same as initial)."""
        cycle = strands.telemetry.metrics.EventLoopCycleMetric(
            event_loop_cycle_id="cycle-1",
            usage=Usage(inputTokens=1000, outputTokens=100, totalTokens=1100),
        )
        invocation = strands.telemetry.metrics.AgentInvocation(cycles=[cycle])
        assert invocation.final_input_tokens == 1000

    def test_context_growth_tokens_single_cycle(self):
        """Test context_growth_tokens with a single cycle (should be 0)."""
        cycle = strands.telemetry.metrics.EventLoopCycleMetric(
            event_loop_cycle_id="cycle-1",
            usage=Usage(inputTokens=1000, outputTokens=100, totalTokens=1100),
        )
        invocation = strands.telemetry.metrics.AgentInvocation(cycles=[cycle])
        assert invocation.context_growth_tokens == 0

    def test_context_size_metrics_multiple_cycles(self):
        """Test context size metrics with multiple cycles (typical tool use scenario)."""
        cycles = [
            strands.telemetry.metrics.EventLoopCycleMetric(
                event_loop_cycle_id="cycle-1",
                usage=Usage(inputTokens=10000, outputTokens=500, totalTokens=10500),
            ),
            strands.telemetry.metrics.EventLoopCycleMetric(
                event_loop_cycle_id="cycle-2",
                usage=Usage(inputTokens=15000, outputTokens=300, totalTokens=15300),
            ),
            strands.telemetry.metrics.EventLoopCycleMetric(
                event_loop_cycle_id="cycle-3",
                usage=Usage(inputTokens=20000, outputTokens=200, totalTokens=20200),
            ),
        ]
        invocation = strands.telemetry.metrics.AgentInvocation(cycles=cycles)

        assert invocation.initial_input_tokens == 10000
        assert invocation.final_input_tokens == 20000
        assert invocation.context_growth_tokens == 10000


class TestEventLoopMetricsContextSizeProperties:
    """Tests for EventLoopMetrics convenience properties for context size metrics."""

    def test_initial_input_tokens_no_invocations(self, event_loop_metrics):
        """Test initial_input_tokens returns 0 when no invocations exist."""
        assert event_loop_metrics.initial_input_tokens == 0

    def test_final_input_tokens_no_invocations(self, event_loop_metrics):
        """Test final_input_tokens returns 0 when no invocations exist."""
        assert event_loop_metrics.final_input_tokens == 0

    def test_context_growth_tokens_no_invocations(self, event_loop_metrics):
        """Test context_growth_tokens returns 0 when no invocations exist."""
        assert event_loop_metrics.context_growth_tokens == 0

    def test_context_size_metrics_with_invocation(self, event_loop_metrics, mock_get_meter_provider):
        """Test context size metrics from EventLoopMetrics with actual invocation."""
        event_loop_metrics.reset_usage_metrics()

        # First cycle - initial context
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-1"})
        event_loop_metrics.update_usage(Usage(inputTokens=10000, outputTokens=500, totalTokens=10500))

        # Second cycle - after tool use
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-2"})
        event_loop_metrics.update_usage(Usage(inputTokens=15000, outputTokens=300, totalTokens=15300))

        # Third cycle - final context
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-3"})
        event_loop_metrics.update_usage(Usage(inputTokens=20000, outputTokens=200, totalTokens=20200))

        assert event_loop_metrics.initial_input_tokens == 10000
        assert event_loop_metrics.final_input_tokens == 20000
        assert event_loop_metrics.context_growth_tokens == 10000

    def test_context_size_metrics_returns_latest_invocation(self, event_loop_metrics, mock_get_meter_provider):
        """Test that context size metrics return values from the latest invocation."""
        # First invocation
        event_loop_metrics.reset_usage_metrics()
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "inv1-cycle-1"})
        event_loop_metrics.update_usage(Usage(inputTokens=5000, outputTokens=100, totalTokens=5100))

        # Second invocation (latest)
        event_loop_metrics.reset_usage_metrics()
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "inv2-cycle-1"})
        event_loop_metrics.update_usage(Usage(inputTokens=8000, outputTokens=200, totalTokens=8200))
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "inv2-cycle-2"})
        event_loop_metrics.update_usage(Usage(inputTokens=12000, outputTokens=150, totalTokens=12150))

        # Should return metrics from latest invocation only
        assert event_loop_metrics.initial_input_tokens == 8000
        assert event_loop_metrics.final_input_tokens == 12000
        assert event_loop_metrics.context_growth_tokens == 4000


class TestGetSummaryContextSizeMetrics:
    """Tests for context size metrics in get_summary() output."""

    def test_get_summary_includes_context_size_metrics(self, event_loop_metrics, mock_get_meter_provider):
        """Test that get_summary includes context size metrics."""
        event_loop_metrics.reset_usage_metrics()
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-1"})
        event_loop_metrics.update_usage(Usage(inputTokens=10000, outputTokens=500, totalTokens=10500))
        event_loop_metrics.start_cycle(attributes={"event_loop_cycle_id": "cycle-2"})
        event_loop_metrics.update_usage(Usage(inputTokens=15000, outputTokens=300, totalTokens=15300))

        summary = event_loop_metrics.get_summary()

        # Check top-level metrics
        assert summary["initial_input_tokens"] == 10000
        assert summary["final_input_tokens"] == 15000
        assert summary["context_growth_tokens"] == 5000

        # Check per-invocation metrics
        assert len(summary["agent_invocations"]) == 1
        invocation_summary = summary["agent_invocations"][0]
        assert invocation_summary["initial_input_tokens"] == 10000
        assert invocation_summary["final_input_tokens"] == 15000
        assert invocation_summary["context_growth_tokens"] == 5000
