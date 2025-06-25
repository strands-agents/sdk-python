from unittest import mock

import pytest

from strands.telemetry import StrandsTelemetry


@pytest.fixture
def mock_tracer_provider():
    with mock.patch("strands.telemetry.config.SDKTracerProvider") as mock_provider:
        yield mock_provider


@pytest.fixture
def mock_get_tracer_provider():
    with mock.patch("strands.telemetry.config.trace_api.get_tracer_provider") as mock_get_tracer_provider:
        mock_provider = mock.MagicMock()
        mock_get_tracer_provider.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def mock_tracer():
    with mock.patch("strands.telemetry.config.trace_api.get_tracer") as mock_get_tracer:
        mock_tracer = mock.MagicMock()
        mock_get_tracer.return_value = mock_tracer
        yield mock_tracer


@pytest.fixture
def mock_set_tracer_provider():
    with mock.patch("strands.telemetry.config.trace_api.set_tracer_provider") as mock_set:
        yield mock_set


@pytest.fixture
def mock_set_global_textmap():
    with mock.patch("strands.telemetry.config.propagate.set_global_textmap") as mock_set_global_textmap:
        yield mock_set_global_textmap


@pytest.fixture
def mock_console_exporter():
    with mock.patch("strands.telemetry.config.ConsoleSpanExporter") as mock_console_exporter:
        yield mock_console_exporter


@pytest.fixture
def mock_otlp_exporter():
    with mock.patch("strands.telemetry.config.OTLPSpanExporter") as mock_otlp_exporter:
        yield mock_otlp_exporter


@pytest.fixture
def mock_batch_processor():
    with mock.patch("strands.telemetry.config.BatchSpanProcessor") as mock_batch_processor:
        yield mock_batch_processor


@pytest.fixture
def mock_simple_processor():
    with mock.patch("strands.telemetry.config.SimpleSpanProcessor") as mock_simple_processor:
        yield mock_simple_processor


@pytest.fixture
def mock_resource():
    with mock.patch("strands.telemetry.config.get_otel_resource") as mock_resource:
        mock_resource_instance = mock.MagicMock()
        mock_resource.return_value = mock_resource_instance
        yield mock_resource


@pytest.fixture
def mock_initialize_tracer():
    with mock.patch("strands.telemetry.StrandsTelemetry._initialize_tracer") as mock_initialize_tracer:
        yield mock_initialize_tracer


def test_init_default(mock_resource, mock_tracer_provider, mock_set_tracer_provider, mock_set_global_textmap):
    """Test initializing the Tracer."""

    StrandsTelemetry()

    mock_resource.assert_called()
    mock_tracer_provider.assert_called_with(resource=mock_resource.return_value)
    mock_set_tracer_provider.assert_called_with(mock_tracer_provider.return_value)
    mock_set_global_textmap.assert_called()


def test_setup_console_exporter(mock_resource, mock_tracer_provider, mock_console_exporter, mock_simple_processor):
    """Test add console exporter"""

    telemetry = StrandsTelemetry()
    # Set the tracer_provider directly
    telemetry.tracer_provider = mock_tracer_provider.return_value
    telemetry.setup_console_exporter()

    mock_console_exporter.assert_called_once()
    mock_simple_processor.assert_called_once_with(mock_console_exporter.return_value)

    mock_tracer_provider.return_value.add_span_processor.assert_called()


def test_setup_otlp_exporter(mock_resource, mock_tracer_provider, mock_otlp_exporter, mock_batch_processor):
    """Test add otlp exporter."""

    telemetry = StrandsTelemetry()
    # Set the tracer_provider directly
    telemetry.tracer_provider = mock_tracer_provider.return_value
    telemetry.setup_otlp_exporter()

    mock_otlp_exporter.assert_called_once()
    mock_batch_processor.assert_called_once_with(mock_otlp_exporter.return_value)

    mock_tracer_provider.return_value.add_span_processor.assert_called()


def test_setup_console_exporter_exception(mock_resource, mock_tracer_provider, mock_console_exporter):
    """Test console exporter with exception."""
    mock_console_exporter.side_effect = Exception("Test exception")

    telemetry = StrandsTelemetry()
    telemetry.tracer_provider = mock_tracer_provider.return_value
    # This should not raise an exception
    telemetry.setup_console_exporter()

    mock_console_exporter.assert_called_once()


def test_setup_otlp_exporter_exception(mock_resource, mock_tracer_provider, mock_otlp_exporter):
    """Test otlp exporter with exception."""
    mock_otlp_exporter.side_effect = Exception("Test exception")

    telemetry = StrandsTelemetry()
    telemetry.tracer_provider = mock_tracer_provider.return_value
    # This should not raise an exception
    telemetry.setup_otlp_exporter()

    mock_otlp_exporter.assert_called_once()
