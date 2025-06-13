import json
import os
from datetime import date, datetime, timezone
from unittest import mock

import pytest
from opentelemetry.trace import (
    StatusCode,  # type: ignore
)

from strands.telemetry.tracer import JSONEncoder, Tracer, get_tracer, serialize
from strands.types.streaming import Usage


@pytest.fixture(autouse=True)
def moto_autouse(moto_env, moto_mock_aws):
    _ = moto_env
    _ = moto_mock_aws


@pytest.fixture
def mock_tracer_provider():
    with mock.patch("strands.telemetry.tracer.SDKTracerProvider") as mock_provider:
        yield mock_provider


@pytest.fixture
def mock_is_initialized():
    with mock.patch("strands.telemetry.tracer.Tracer._is_initialized") as mock_is_initialized:
        yield mock_is_initialized


@pytest.fixture
def mock_get_tracer_provider():
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer_provider") as mock_get_tracer_provider:
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
def mock_set_tracer_provider():
    with mock.patch("strands.telemetry.tracer.trace_api.set_tracer_provider") as mock_set:
        yield mock_set


@pytest.fixture
def mock_otlp_exporter():
    with mock.patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter") as mock_exporter:
        yield mock_exporter


@pytest.fixture
def mock_console_exporter():
    with mock.patch("strands.telemetry.tracer.ConsoleSpanExporter") as mock_exporter:
        yield mock_exporter


@pytest.fixture
def mock_resource():
    with mock.patch("strands.telemetry.tracer.Resource") as mock_resource:
        yield mock_resource


@pytest.fixture
def clean_env():
    """Fixture to provide a clean environment for each test."""
    with mock.patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def env_with_otlp():
    """Fixture with OTLP environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://env-endpoint",
        },
    ):
        yield


@pytest.fixture
def env_with_console():
    """Fixture with console export environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "STRANDS_OTEL_ENABLE_CONSOLE_EXPORT": "true",
        },
    ):
        yield


@pytest.fixture
def env_with_both():
    """Fixture with both OTLP and console export environment variables."""
    with mock.patch.dict(
        os.environ,
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://env-endpoint",
            "STRANDS_OTEL_ENABLE_CONSOLE_EXPORT": "true",
        },
    ):
        yield


@pytest.fixture
def mock_initialize():
    with mock.patch("strands.telemetry.tracer.Tracer._initialize_tracer") as mock_initialize:
        yield mock_initialize


def test_init_default(mock_is_initialized, mock_get_tracer_provider):
    """Test initializing the Tracer with default parameters."""
    mock_is_initialized.return_value = False
    mock_get_tracer_provider.return_value = None

    tracer = Tracer()

    assert tracer.service_name == "strands-agents"
    assert tracer.otlp_endpoint is None
    assert tracer.otlp_headers == {}
    assert tracer.enable_console_export is False
    assert tracer.tracer_provider is None
    assert tracer.tracer is None


def test_init_with_env_endpoint():
    """Test initializing the Tracer with endpoint from environment variable."""
    with mock.patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://test-endpoint"}):
        tracer = Tracer()

        assert tracer.otlp_endpoint == "http://test-endpoint"


def test_init_with_env_console_export():
    """Test initializing the Tracer with console export from environment variable."""
    with mock.patch.dict(os.environ, {"STRANDS_OTEL_ENABLE_CONSOLE_EXPORT": "true"}):
        tracer = Tracer()

        assert tracer.enable_console_export is True


def test_init_with_env_headers():
    """Test initializing the Tracer with headers from environment variable."""
    with mock.patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_HEADERS": "key1=value1,key2=value2"}):
        tracer = Tracer()

        assert tracer.otlp_headers == {"key1": "value1", "key2": "value2"}


def test_initialize_tracer_with_console(
    mock_is_initialized,
    mock_tracer_provider,
    mock_set_tracer_provider,
    mock_console_exporter,
    mock_resource,
):
    """Test initializing the tracer with console exporter."""
    mock_is_initialized.return_value = False
    mock_resource_instance = mock.MagicMock()
    mock_resource.create.return_value = mock_resource_instance

    # Initialize Tracer
    Tracer(enable_console_export=True)

    # Verify the tracer provider was created with correct resource
    mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)

    # Verify console exporter was added
    mock_console_exporter.assert_called_once()
    mock_tracer_provider.return_value.add_span_processor.assert_called_once()

    # Verify set_tracer_provider was called
    mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider.return_value)


def test_initialize_tracer_with_otlp(
    mock_is_initialized, mock_tracer_provider, mock_set_tracer_provider, mock_otlp_exporter, mock_resource
):
    """Test initializing the tracer with OTLP exporter."""
    mock_is_initialized.return_value = False

    mock_resource_instance = mock.MagicMock()
    mock_resource.create.return_value = mock_resource_instance

    # Initialize Tracer
    Tracer(otlp_endpoint="http://test-endpoint")

    # Verify the tracer provider was created with correct resource
    mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)

    # Verify OTLP exporter was added with correct endpoint
    mock_otlp_exporter.assert_called_once()
    assert "endpoint" in mock_otlp_exporter.call_args.kwargs
    assert "headers" in mock_otlp_exporter.call_args.kwargs

    # Verify set_tracer_provider was called
    mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider.return_value)


def test_start_span_no_tracer():
    """Test starting a span when no tracer is configured."""
    tracer = Tracer()
    span = tracer._start_span("test_span")

    assert span is None


def test_start_span(mock_tracer):
    """Test starting a span with attributes."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        span = tracer._start_span("test_span", attributes={"key": "value"})

        mock_tracer.start_span.assert_called_once_with(name="test_span", context=None)
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
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        model_id = "test-model"

        span = tracer.start_model_invoke_span(agent_name="TestAgent", messages=messages, model_id=model_id)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "Model invoke"
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("agent.name", "TestAgent")
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", model_id)
        assert span is not None


def test_end_model_invoke_span(mock_span):
    """Test ending a model invoke span."""
    tracer = Tracer()
    message = {"role": "assistant", "content": [{"text": "Response"}]}
    usage = Usage(inputTokens=10, outputTokens=20, totalTokens=30)

    tracer.end_model_invoke_span(mock_span, message, usage)

    mock_span.set_attribute.assert_any_call("gen_ai.completion", json.dumps(message["content"]))
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 10)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 20)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 30)
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_tool_call_span(mock_tracer):
    """Test starting a tool call span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        tool = {"name": "test-tool", "toolUseId": "123", "input": {"param": "value"}}

        span = tracer.start_tool_call_span(tool)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "Tool: test-tool"
        mock_span.set_attribute.assert_any_call(
            "gen_ai.prompt", json.dumps({"name": "test-tool", "toolUseId": "123", "input": {"param": "value"}})
        )
        mock_span.set_attribute.assert_any_call("tool.name", "test-tool")
        mock_span.set_attribute.assert_any_call("tool.id", "123")
        mock_span.set_attribute.assert_any_call("tool.parameters", json.dumps({"param": "value"}))
        assert span is not None


def test_end_tool_call_span(mock_span):
    """Test ending a tool call span."""
    tracer = Tracer()
    tool_result = {"status": "success", "content": [{"text": "Tool result"}]}

    tracer.end_tool_call_span(mock_span, tool_result)

    mock_span.set_attribute.assert_any_call("tool.result", json.dumps(tool_result.get("content")))
    mock_span.set_attribute.assert_any_call("gen_ai.completion", json.dumps(tool_result.get("content")))
    mock_span.set_attribute.assert_any_call("tool.status", "success")
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_event_loop_cycle_span(mock_tracer):
    """Test starting an event loop cycle span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        event_loop_kwargs = {"event_loop_cycle_id": "cycle-123"}
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        span = tracer.start_event_loop_cycle_span(event_loop_kwargs, messages=messages)

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "Cycle cycle-123"
        mock_span.set_attribute.assert_any_call("gen_ai.prompt", json.dumps(messages))
        mock_span.set_attribute.assert_any_call("event_loop.cycle_id", "cycle-123")
        assert span is not None


def test_end_event_loop_cycle_span(mock_span):
    """Test ending an event loop cycle span."""
    tracer = Tracer()
    message = {"role": "assistant", "content": [{"text": "Response"}]}
    tool_result_message = {"role": "assistant", "content": [{"toolResult": {"response": "Success"}}]}

    tracer.end_event_loop_cycle_span(mock_span, message, tool_result_message)

    mock_span.set_attribute.assert_any_call("gen_ai.completion", json.dumps(message["content"]))
    mock_span.set_attribute.assert_any_call("tool.result", json.dumps(tool_result_message["content"]))
    mock_span.set_status.assert_called_once_with(StatusCode.OK)
    mock_span.end.assert_called_once()


def test_start_agent_span(mock_tracer):
    """Test starting an agent span."""
    with mock.patch("strands.telemetry.tracer.trace_api.get_tracer", return_value=mock_tracer):
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        prompt = "What's the weather today?"
        model_id = "test-model"
        tools = [{"name": "weather_tool"}]
        custom_attrs = {"custom_attr": "value"}

        span = tracer.start_agent_span(
            prompt=prompt,
            agent_name="WeatherAgent",
            model_id=model_id,
            tools=tools,
            custom_trace_attributes=custom_attrs,
        )

        mock_tracer.start_span.assert_called_once()
        assert mock_tracer.start_span.call_args[1]["name"] == "WeatherAgent"
        mock_span.set_attribute.assert_any_call("gen_ai.system", "strands-agents")
        mock_span.set_attribute.assert_any_call("agent.name", "WeatherAgent")
        mock_span.set_attribute.assert_any_call("gen_ai.prompt", prompt)
        mock_span.set_attribute.assert_any_call("gen_ai.request.model", model_id)
        mock_span.set_attribute.assert_any_call("custom_attr", "value")
        assert span is not None


def test_end_agent_span(mock_span):
    """Test ending an agent span."""
    tracer = Tracer()

    # Mock AgentResult with metrics
    mock_metrics = mock.MagicMock()
    mock_metrics.accumulated_usage = {"inputTokens": 50, "outputTokens": 100, "totalTokens": 150}

    mock_response = mock.MagicMock()
    mock_response.metrics = mock_metrics
    mock_response.__str__ = mock.MagicMock(return_value="Agent response")

    tracer.end_agent_span(mock_span, mock_response)

    mock_span.set_attribute.assert_any_call("gen_ai.completion", "Agent response")
    mock_span.set_attribute.assert_any_call("gen_ai.usage.prompt_tokens", 50)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.completion_tokens", 100)
    mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
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
        tracer2 = get_tracer(otlp_endpoint="http://new-endpoint")

        assert tracer1 is not tracer2
        assert tracer2.otlp_endpoint == "http://new-endpoint"


def test_get_tracer_parameters():
    """Test that get_tracer passes parameters correctly."""
    # Reset the singleton first
    with mock.patch("strands.telemetry.tracer._tracer_instance", None):
        tracer = get_tracer(
            service_name="test-service",
            otlp_endpoint="http://test-endpoint",
            otlp_headers={"key": "value"},
            enable_console_export=True,
        )

        assert tracer.service_name == "test-service"
        assert tracer.otlp_endpoint == "http://test-endpoint"
        assert tracer.otlp_headers == {"key": "value"}
        assert tracer.enable_console_export is True


def test_initialize_tracer_with_invalid_otlp_endpoint(
    mock_is_initialized, mock_tracer_provider, mock_set_tracer_provider, mock_otlp_exporter, mock_resource
):
    """Test initializing the tracer with an invalid OTLP endpoint."""
    mock_is_initialized.return_value = False

    mock_resource_instance = mock.MagicMock()
    mock_resource.create.return_value = mock_resource_instance
    mock_otlp_exporter.side_effect = Exception("Connection error")

    # This should not raise an exception, but should log an error

    # Initialize Tracer
    Tracer(otlp_endpoint="http://invalid-endpoint")

    # Verify the tracer provider was created with correct resource
    mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)

    # Verify OTLP exporter was attempted
    mock_otlp_exporter.assert_called_once()

    # Verify set_tracer_provider was still called
    mock_set_tracer_provider.assert_called_once_with(mock_tracer_provider.return_value)


def test_initialize_tracer_with_custom_tracer_provider(mock_get_tracer_provider, mock_resource):
    """Test initializing the tracer with NoOpTracerProvider."""
    mock_is_initialized.return_value = True
    tracer = Tracer(otlp_endpoint="http://invalid-endpoint")

    mock_get_tracer_provider.assert_called()
    mock_resource.assert_not_called()

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


def test_force_flush_with_error(mock_span, mock_tracer_provider):
    """Test force flush with error handling."""
    # Setup the tracer with a provider that raises an exception on force_flush
    tracer = Tracer()
    tracer.tracer_provider = mock_tracer_provider
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
        tracer = Tracer(enable_console_export=True)
        tracer.tracer = mock_tracer

        mock_span = mock.MagicMock()
        parent_span = mock.MagicMock()
        mock_tracer.start_span.return_value = mock_span

        span = tracer.start_model_invoke_span(parent_span=parent_span, agent_name="TestAgent", model_id="test-model")

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


def test_init_with_no_env_or_param(clean_env):
    """Test initializing with neither environment variable nor constructor parameter."""
    tracer = Tracer()
    assert tracer.otlp_endpoint is None
    assert tracer.enable_console_export is False

    tracer = Tracer(otlp_endpoint="http://param-endpoint")
    assert tracer.otlp_endpoint == "http://param-endpoint"

    tracer = Tracer(enable_console_export=True)
    assert tracer.enable_console_export is True


def test_constructor_params_with_otlp_env(env_with_otlp):
    """Test constructor parameters precedence over OTLP environment variable."""
    # Constructor parameter should take precedence
    tracer = Tracer(otlp_endpoint="http://constructor-endpoint")
    assert tracer.otlp_endpoint == "http://constructor-endpoint"

    # Without constructor parameter, should use env var
    tracer = Tracer()
    assert tracer.otlp_endpoint == "http://env-endpoint"


def test_constructor_params_with_console_env(env_with_console):
    """Test constructor parameters precedence over console environment variable."""
    # Constructor parameter should take precedence
    tracer = Tracer(enable_console_export=False)
    assert tracer.enable_console_export is False

    # Without explicit constructor parameter, should use env var
    tracer = Tracer()
    assert tracer.enable_console_export is True


def test_fallback_to_env_vars(env_with_both):
    """Test fallback to environment variables when no constructor parameters."""
    tracer = Tracer()
    assert tracer.otlp_endpoint == "http://env-endpoint"
    assert tracer.enable_console_export is True

    # Constructor parameters should still take precedence
    tracer = Tracer(otlp_endpoint="http://constructor-endpoint", enable_console_export=False)
    assert tracer.otlp_endpoint == "http://constructor-endpoint"
    assert tracer.enable_console_export is False
