"""Telemetry module.

This module provides metrics and tracing functionality.
"""

from .config import StrandsTelemetry, get_otel_resource
from .metrics import EventLoopMetrics, MetricsClient, Trace, metrics_to_string
from .tracer import Tracer, get_tracer

__all__ = [
    # Metrics
    "EventLoopMetrics",
    "Trace",
    "metrics_to_string",
    "MetricsClient",
    # Tracer
    "Tracer",
    "get_tracer",
    # Resource
    "get_otel_resource",
    # Telemetry Setup
    "StrandsTelemetry",
]
