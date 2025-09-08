import json
import os
from datetime import date, datetime, timezone
from unittest import mock

import pytest
from opentelemetry.trace import (
    SpanKind,
    StatusCode,  # type: ignore
)

from strands.telemetry.tracer import JSONEncoder, Tracer, get_tracer, serialize
from strands.types.content import ContentBlock
from strands.types.streaming import StopReason, Usage


@pytest.fixture(autouse=True)
def moto_autouse(moto_env, moto_mock_aws):
    _ = moto_env
    _ = moto_mock_aws


@pytest.fixture
def mock_get_tracer_provider():
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer_provider") as mock_get_tracer_provider:
        mock_tracer = mock.MagicMock()
        mock_get_tracer_provider.get_tracer.return_value = mock_tracer
        yield mock_get_tracer_provider


@pytest.fixture
def mock_tracer():
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer") as mock_get_tracer:
        mock_tracer = mock.MagicMock()
        mock_get_tracer.return_value = mock_tracer
        yield mock_tracer


@pytest.fixture
def mock_span():
    mock_span = mock.MagicMock()
    return mock_span


@pytest.fixture
def clean_env():
    """Fixture to provide a clean environment for each test."""
    with mock.patch.dict(os.environ, {}, clear=True):
        yield


def test_init_default():
    """Test initializing the Tracer with default parameters."""
    tracer = Tracer()

    assert tracer.service_name == "strands.telemetry.tracer"
    assert tracer.tracer_provider is not None
    assert tracer.tracer is not None


def test_start_span_no_tracer():
    """Test starting a span when no tracer is configured."""
    tracer = Tracer()
    span = tracer._start_span("test_span")

    assert span is not None


def test_start_span(mock_tracer):
    """Test starting a span with attributes."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        span = tracer._start_span("test_span", attributes={"key": "value"})

        mock_tracer.start_span.assert_called_once_with(name="test_span", context=None, kind=SpanKind.INTERNAL)
        mock_span.set_attribute.assert_any_call("key", "value")
        assert span is not None


def test_set_attributes(mock_span):
    """Test setting attributes on a span."""
    tracer = Tracer()
    attributes = {"str_attr": "value", "int_attr": 123, "bool_attr": True}

    tracer._set_attributes(mock_span, attributes)

    # Check that set_attribute was called for each attribute
    calls = [mock.call(k, v) for k, v in attributes.items()]
    mock_span.set_attribute.assert_has_calls(calls, any_order=True)


def test_end_span_no_span():
    """Test ending a span when span is None."""
    tracer = Tracer()
    # Should not raise an exception
    tracer._end_span(None)


def test_end_span(mock_span):
    """Test ending a span with attributes and no error."""
    tracer = Tracer()
    attributes = {"key": "value"}

    tracer._end_span(mock_span, attributes)

    mock_span.set_attribute.assert_any_call("key", "value")
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_end_span_with_error(mock_span):
    """Test ending a span with an error."""
    tracer = Tracer()
    error = Exception("Test error")

    tracer._end_span(mock_span, error=error)

    mock_span.set_status.assert_called_once_with(StatusCode.ERROR, str(error))
    mock_span.record_exception.assert_called_once_with(error)
    mock_span.end.assert_called_once()


def test_end_span_with_error_message(mock_span):
    """Test ending a span with an error message."""
    tracer = Tracer()
    error_message = "Test error message"

    tracer.end_span_with_error(mock_span, error_message)

    mock_span.set_status.assert_called_once()
    assert mock_span.set_status.call_args[0][0] == StatusCode.ERROR
    mock_span.end.assert_called_once()


def test_start_model_invoke_span(mock_tracer):
    """Test starting a model invoke span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        model_id = "test-model"

        span = tracer.start_model_invoke_span(messages=messages, agent_name="TestAgent", model_id=model_id)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "chat"
        assert mock_tracer.start_span.call_args[1]["kind"] == SpanKind.CLIENT
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "chat")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", model_id)
        mock_span.add_event.assert_called_with(
            "gen_ai.user.message", attributes={"content": json.dumps(messages[0]["content"])}
        )
        assert span is not None


def test_end_model_invoke_span(mock_span):
    """Test ending a model invoke span."""
    tracer = Tracer()
    message = {"role": "assistant", "content": [{"text": "Response"}]}
    usage = Usage(inputTokens=10, outputTokens=20, totalTokens=30)
    stop_reason: StopReason = "end_turn"

    tracer.end_model_invoke_span(mock_span, message, usage, stop_reason)

    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 20)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 20)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 30)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_read_input_tokens", 0)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_write_input_tokens", 0)
    mock_span.add_event.assert_called_with(
        "gen_ai.choice",
        attributes={"message": json.dumps(message["content"]), "finish_reason": "end_turn"},
    )
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_tool_call_span(mock_tracer):
    """Test starting a tool call span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        tool = {"name": "test-tool", "toolUseId": "123", "input": {"param": "value"}}

        span = tracer.start_tool_call_span(tool)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "execute_tool test-tool"
        mock_span.set_attribute.assert_any_call("gen_ai.tool.name", "test-tool")
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "execute_tool")
        mock_span.set_attribute.assert_any_call("gen_ai.tool.call.id", "123")
        mock_span.add_event.assert_any_call(
            "gen_ai.tool.message", attributes={"role": "tool", "content": json.dumps({"param": "value"}), "id": "123"}
        )
        assert span is not None


def test_start_swarm_call_span_with_string_task(mock_tracer):
    """Test starting a swarm call span with task as string."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        task = "Design foo bar"

        span = tracer.start_multiagent_span(task, "swarm")

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "invoke_swarm"
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.agent.name", "swarm")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "invoke_swarm")
        mock_span.add_event.assert_any_call("gen_ai.user.message", attributes={"content": "Design foo bar"})
        assert span is not None


def test_start_swarm_span_with_contentblock_task(mock_tracer):
    """Test starting a swarm call span with task as list of contentBlock."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        task = [ContentBlock(text="Original Task: foo bar")]

        span = tracer.start_multiagent_span(task, "swarm")

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "invoke_swarm"
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.agent.name", "swarm")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "invoke_swarm")
        mock_span.add_event.assert_any_call(
            "gen_ai.user.message", attributes={"content": '[{"text": "Original Task: foo bar"}]'}
        )
        assert span is not None


def test_end_swarm_span(mock_span):
    """Test ending a tool call span."""
    tracer = Tracer()
    swarm_final_reuslt = "foo bar bar"

    tracer.end_swarm_span(mock_span, swarm_final_reuslt)

    mock_span.add_event.assert_called_with(
        "gen_ai.choice",
        attributes={"message": "foo bar bar"},
    )


def test_start_graph_call_span(mock_tracer):
    """Test starting a graph call span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        tool = {"name": "test-tool", "toolUseId": "123", "input": {"param": "value"}}

        span = tracer.start_tool_call_span(tool)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "execute_tool test-tool"
        mock_span.set_attribute.assert_any_call("gen_ai.tool.name", "test-tool")
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.operation.name", "execute_tool")
        mock_span.set_attribute.assert_any_call("gen_ai.tool.call.id", "123")
        mock_span.add_event.assert_any_call(
            "gen_ai.tool.message", attributes={"role": "tool", "content": json.dumps({"param": "value"}), "id": "123"}
        )
        assert span is not None


def test_end_tool_call_span(mock_span):
    """Test ending a tool call span."""
    tracer = Tracer()
    tool_result = {"status": "success", "content": [{"text": "Tool result"}]}

    tracer.end_tool_call_span(mock_span, tool_result)

    mock_span.set_attribute.assert_any_call("tool.status", "success")
    mock_span.add_event.assert_called_with(
        "gen_ai.choice",
        attributes={"message": json.dumps(tool_result.get("content")), "id": ""},
    )
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_event_loop_cycle_span(mock_tracer):
    """Test starting an event loop cycle span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        event_loop_kwargs = {"event_loop_cycle_id": "cycle-123"}
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        span = tracer.start_event_loop_cycle_span(event_loop_kwargs, messages=messages)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "execute_event_loop_cycle"
        mock_span.set_attribute.assert_any_call("event_loop.cycle_id", "cycle-123")
        mock_span.add_event.assert_any_call(
            "gen_ai.user.message", attributes={"content": json.dumps([{"text": "Hello"}])}
        )
        assert span is not None


def test_end_event_loop_cycle_span(mock_span):
    """Test ending an event loop cycle span."""
    tracer = Tracer()
    message = {"role": "assistant", "content": [{"text": "Response"}]}
    tool_result_message = {"role": "assistant", "content": [{"toolResult": {"response": "Success"}}]}

    tracer.end_event_loop_cycle_span(mock_span, message, tool_result_message)

    mock_span.add_event.assert_called_with(
        "gen_ai.choice",
        attributes={
            "message": json.dumps(message["content"]),
            "tool.result": json.dumps(tool_result_message["content"]),
        },
    )
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_agent_span(mock_tracer):
    """Test starting an agent span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        content = [{"text": "test prompt"}]
        model_id = "test-model"
        tools = [{"name": "weather_tool"}]
        custom_attrs = {"custom_attr": "value"}

        span = tracer.start_agent_span(
            custom_trace_attributes=custom_attrs,
            agent_name="WeatherAgent",
            messages=[{"content": content, "role": "user"}],
            model_id=model_id,
            tools=tools,
        )

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "invoke_agent WeatherAgent"
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("gen_ai.agent.name", "WeatherAgent")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", model_id)
        mock_span.set_attribute.assert_any_call("custom_attr", "value")
        mock_span.add_event.assert_any_call("gen_ai.user.message", attributes={"content": json.dumps(content)})
        assert span is not None


def test_end_agent_span(mock_span):
    """Test ending an agent span."""
    tracer = Tracer()

    # Mock AgentResult with metrics
    mock_metrics = mock.MagicMock()
    mock_metrics.accumulated_usage = {"inputTokens": 50, "outputTokens": 100, "totalTokens": 150}

    mock_response = mock.MagicMock()
    mock_response.metrics = mock_metrics
    mock_response.stop_reason = "end_turn"
    mock_response.__str__ = mock.MagicMock(return_value="Agent response")

    tracer.end_agent_span(mock_span, mock_response)

    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_read_input_tokens", 0)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_write_input_tokens", 0)
    mock_span.add_event.assert_any_call(
        "gen_ai.choice",
        attributes={"message": "Agent response", "finish_reason": "end_turn"},
    )
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_end_model_invoke_span_with_cache_metrics(mock_span):
    """Test ending a model invoke span with cache metrics."""
    tracer = Tracer()
    message = {"role": "assistant", "content": [{"text": "Response"}]}
    usage = Usage(
        inputTokens=10,
        outputTokens=20,
        totalTokens=30,
        cacheReadInputTokens=5,
        cacheWriteInputTokens=3,
    )
    stop_reason: StopReason = "end_turn"

    tracer.end_model_invoke_span(mock_span, message, usage, stop_reason)

    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 20)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 20)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 30)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_read_input_tokens", 5)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_write_input_tokens", 3)
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_end_agent_span_with_cache_metrics(mock_span):
    """Test ending an agent span with cache metrics."""
    tracer = Tracer()

    # Mock AgentResult with metrics including cache tokens
    mock_metrics = mock.MagicMock()
    mock_metrics.accumulated_usage = {
        "inputTokens": 50,
        "outputTokens": 100,
        "totalTokens": 150,
        "cacheReadInputTokens": 25,
        "cacheWriteInputTokens": 10,
    }

    mock_response = mock.MagicMock()
    mock_response.metrics = mock_metrics
    mock_response.stop_reason = "end_turn"
    mock_response.__str__ = mock.MagicMock(return_value="Agent response")

    tracer.end_agent_span(mock_span, mock_response)

    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_read_input_tokens", 25)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.cache_write_input_tokens", 10)
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_get_tracer_singleton():
    """Test that get_tracer returns a singleton instance."""
    # Reset the singleton first
    with mock.patch("strands.telemetry.tracer._tracer_instance", None):
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2


def test_get_tracer_new_endpoint():
    """Test that get_tracer creates a new instance when endpoint changes."""
    # Reset the singleton first
    with mock.patch("strands.telemetry.tracer._tracer_instance", None):
        tracer1 = get_tracer()
        tracer2 = get_tracer()

        assert tracer1 is tracer2


def test_initialize_tracer_with_custom_tracer_provider(mock_get_tracer_provider):
    """Test initializing the tracer with NoOpTracerProvider."""
    tracer = Tracer()

    mock_get_tracer_provider.assert_called()

    assert tracer.tracer_provider is not None
    assert tracer.tracer is not None


def test_end_span_with_exception_handling(mock_span):
    """Test ending a span with exception handling."""
    tracer = Tracer()

    # Make set_attribute throw an exception
    mock_span.set_attribute.side_effect = Exception("Test error during set_attribute")

    try:
        # Should not raise an exception
        tracer._end_span(mock_span, {"key": "value"})

        # Should still try to end the span
        mock_span.end.assert_called_once()
    except Exception:
        pytest.fail("_end_span should not raise exceptions")


def test_force_flush_with_error(mock_span, mock_get_tracer_provider):
    """Test force flush with error handling."""
    # Setup the tracer with a provider that raises an exception on force_flush
    tracer = Tracer()

    mock_tracer_provider = mock_get_tracer_provider.return_value
    mock_tracer_provider.force_flush.side_effect = Exception("Force flush error")

    # Should not raise an exception
    tracer._end_span(mock_span)

    # Verify force_flush was called
    mock_tracer_provider.force_flush.assert_called_once()


def test_end_tool_call_span_with_none(mock_span):
    """Test ending a tool call span with None result."""
    tracer = Tracer()

    # Should not raise an exception
    tracer.end_tool_call_span(mock_span, None)

    # Should still end the span
    mock_span.end.assert_called_once()


def test_start_model_invoke_span_with_parent(mock_tracer):
    """Test starting a model invoke span with a parent span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer()
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        parent_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        span = tracer.start_model_invoke_span(
            messages=[], parent_span=parent_span, agent_name="TestAgent", model_id="test-model"
        )

        # Verify trace.set_span_in_context was called with parent span
        mock_tracer.start_span.assert_called_once()

        # Verify span was returned
        assert span is mock_span


@pytest.mark.parametrize(
    "input_data, expected_result",
    [
        ("test string", '"test string"'),
        (1234, "1234"),
        (13.37, "13.37"),
        (False, "false"),
        (None, "null"),
    ],
)
def test_json_encoder_serializable(input_data, expected_result):
    """Test encoding of serializable values."""
    encoder = JSONEncoder()

    result = encoder.encode(input_data)
    assert result == expected_result


def test_json_encoder_datetime():
    """Test encoding datetime and date objects."""
    encoder = JSONEncoder()

    dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = encoder.encode(dt)
    assert result == f'"{dt.isoformat()}"'

    d = date(2025, 1, 1)
    result = encoder.encode(d)
    assert result == f'"{d.isoformat()}"'


def test_json_encoder_list():
    """Test encoding a list with mixed content."""
    encoder = JSONEncoder()

    non_serializable = lambda x: x  # noqa: E731

    data = ["value", 42, 13.37, non_serializable, None, {"key": True}, ["value here"]]

    result = json.loads(encoder.encode(data))
    assert result == ["value", 42, 13.37, "<replaced>", None, {"key": True}, ["value here"]]


def test_json_encoder_dict():
    """Test encoding a dict with mixed content."""
    encoder = JSONEncoder()

    class UnserializableClass:
        def __str__(self):
            return "Unserializable Object"

    non_serializable = lambda x: x  # noqa: E731

    now = datetime.now(timezone.utc)

    data = {
        "metadata": {
            "timestamp": now,
            "version": "1.0",
            "debug_info": {"object": non_serializable, "callable": lambda x: x + 1},  # noqa: E731
        },
        "content": [
            {"type": "text", "value": "Hello world"},
            {"type": "binary", "value": non_serializable},
            {"type": "mixed", "values": [1, "text", non_serializable, {"nested": non_serializable}]},
        ],
        "statistics": {
            "processed": 100,
            "failed": 5,
            "details": [{"id": 1, "status": "ok"}, {"id": 2, "status": "error", "error_obj": non_serializable}],
        },
        "list": [
            non_serializable,
            1234,
            13.37,
            True,
            None,
            "string here",
        ],
    }

    expected = {
        "metadata": {
            "timestamp": now.isoformat(),
            "version": "1.0",
            "debug_info": {"object": "<replaced>", "callable": "<replaced>"},
        },
        "content": [
            {"type": "text", "value": "Hello world"},
            {"type": "binary", "value": "<replaced>"},
            {"type": "mixed", "values": [1, "text", "<replaced>", {"nested": "<replaced>"}]},
        ],
        "statistics": {
            "processed": 100,
            "failed": 5,
            "details": [{"id": 1, "status": "ok"}, {"id": 2, "status": "error", "error_obj": "<replaced>"}],
        },
        "list": [
            "<replaced>",
            1234,
            13.37,
            True,
            None,
            "string here",
        ],
    }

    result = json.loads(encoder.encode(data))

    assert result == expected


def test_json_encoder_value_error():
    """Test encoding values that cause ValueError."""
    encoder = JSONEncoder()

    # A very large integer that exceeds JSON limits and throws ValueError
    huge_number = 2**100000

    # Test in a dictionary
    dict_data = {"normal": 42, "huge": huge_number}
    result = json.loads(encoder.encode(dict_data))
    assert result == {"normal": 42, "huge": "<replaced>"}

    # Test in a list
    list_data = [42, huge_number]
    result = json.loads(encoder.encode(list_data))
    assert result == [42, "<replaced>"]

    # Test just the value
    result = json.loads(encoder.encode(huge_number))
    assert result == "<replaced>"


def test_serialize_non_ascii_characters():
    """Test that non-ASCII characters are preserved in JSON serialization."""

    # Test with Japanese text
    japanese_text = "こんにちは世界"
    result = serialize({"text": japanese_text})
    assert japanese_text in result
    assert "\\u" not in result

    # Test with emoji
    emoji_text = "Hello 🌍"
    result = serialize({"text": emoji_text})
    assert emoji_text in result
    assert "\\u" not in result

    # Test with Chinese characters
    chinese_text = "你好，世界"
    result = serialize({"text": chinese_text})
    assert chinese_text in result
    assert "\\u" not in result

    # Test with mixed content
    mixed_text = {"ja": "こんにちは", "emoji": "😊", "zh": "你好", "en": "hello"}
    result = serialize(mixed_text)
    assert "こんにちは" in result
    assert "😊" in result
    assert "你好" in result
    assert "\\u" not in result


def test_serialize_vs_json_dumps():
    """Test that serialize behaves differently from default json.dumps for non-ASCII characters."""

    # Test with Japanese text
    japanese_text = "こんにちは世界"

    # Default json.dumps should escape non-ASCII characters
    default_result = json.dumps({"text": japanese_text})
    assert "\\u" in default_result

    # Our serialize function should preserve non-ASCII characters
    custom_result = serialize({"text": japanese_text})
    assert japanese_text in custom_result
    assert "\\u" not in custom_result
