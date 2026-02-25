"""Telemetry module.

This module provides metrics, tracing, and evaluation telemetry functionality.
"""

from .config import StrandsTelemetry
from .evaluation import (
    EvaluationEventEmitter,
    EvaluationResult,
    add_evaluation_event,
    set_test_case_context,
    set_test_suite_context,
)
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
    # Telemetry Setup
    "StrandsTelemetry",
    # Evaluation
    "EvaluationResult",
    "EvaluationEventEmitter",
    "add_evaluation_event",
    "set_test_suite_context",
    "set_test_case_context",
]
