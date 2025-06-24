"""Telemetry module.

This module provides metrics and tracing functionality.
"""

from .config import get_otel_resource
from .metrics import EventLoopMetrics, MetricsClient, Trace, metrics_to_string
from .tracer import Tracer, get_tracer

__all__ = [
    "EventLoopMetrics",
    "Trace",
    "metrics_to_string",
    "Tracer",
    "get_tracer",
    "MetricsClient",
    "get_otel_resource",
]
