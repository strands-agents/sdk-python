"""GenAI evaluation result telemetry.

This module provides components for emitting GenAI evaluation results as
OpenTelemetry events following the ``gen_ai.evaluation.result`` semantic
convention. The feature is opt-in — developers explicitly call the APIs
to emit evaluation events on spans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from opentelemetry.trace import Span

from ..types.traces import AttributeValue

logger = logging.getLogger(__name__)

_EVENT_NAME = "gen_ai.evaluation.result"


@dataclass
class EvaluationResult:
    """A GenAI evaluation result following OTel semantic conventions.

    Represents a single evaluation outcome that can be emitted as a
    ``gen_ai.evaluation.result`` event on an OTel span.

    Args:
        name: Evaluation metric name (Required). Maps to ``gen_ai.evaluation.name``.
        score_value: Numeric evaluation score. Maps to ``gen_ai.evaluation.score.value``.
        score_label: Human-readable score label. Maps to ``gen_ai.evaluation.score.label``.
        explanation: Free-form explanation. Maps to ``gen_ai.evaluation.explanation``.
        response_id: Completion ID for correlation. Maps to ``gen_ai.response.id``.
        error_type: Error class description. Maps to ``error.type``.
    """

    name: str
    score_value: float | None = None
    score_label: str | None = None
    explanation: str | None = None
    response_id: str | None = None
    error_type: str | None = None

    def to_otel_attributes(self) -> dict[str, AttributeValue]:
        """Convert to OTel event attributes, omitting None fields.

        Returns:
            Dictionary mapping OTel attribute names to their values. The
            ``gen_ai.evaluation.name`` key is always present. Optional fields
            are included only when their value is not None.
        """
        attrs: dict[str, AttributeValue] = {"gen_ai.evaluation.name": self.name}
        if self.score_value is not None:
            attrs["gen_ai.evaluation.score.value"] = self.score_value
        if self.score_label is not None:
            attrs["gen_ai.evaluation.score.label"] = self.score_label
        if self.explanation is not None:
            attrs["gen_ai.evaluation.explanation"] = self.explanation
        if self.response_id is not None:
            attrs["gen_ai.response.id"] = self.response_id
        if self.error_type is not None:
            attrs["error.type"] = self.error_type
        return attrs


class EvaluationEventEmitter:
    """Emits gen_ai.evaluation.result events on OTel spans."""

    @staticmethod
    def emit(span: Span | None, result: EvaluationResult) -> None:
        """Emit an evaluation result event on the given span.

        Args:
            span: Target OTel span. Skipped if None or not recording.
            result: The evaluation result to emit.
        """
        if span is None or not span.is_recording():
            return
        span.add_event(_EVENT_NAME, attributes=result.to_otel_attributes())


def add_evaluation_event(
    span: Span | None,
    result: EvaluationResult | None = None,
    *,
    name: str | None = None,
    score_value: float | None = None,
    score_label: str | None = None,
    explanation: str | None = None,
    response_id: str | None = None,
    error_type: str | None = None,
) -> None:
    """Emit a gen_ai.evaluation.result event on a span.

    Accepts either an ``EvaluationResult`` instance or keyword arguments.
    When ``result`` is provided it takes precedence over keyword arguments.

    Args:
        span: Target OTel span. Skipped if None or not recording.
        result: Pre-built EvaluationResult. Takes precedence over kwargs.
        name: Evaluation metric name (required if result is None).
        score_value: Numeric score.
        score_label: Human-readable label.
        explanation: Free-form explanation.
        response_id: Completion ID for correlation.
        error_type: Error class description.

    Raises:
        ValueError: If neither result nor name is provided.
    """
    if result is None:
        if name is None:
            raise ValueError("Either 'result' or 'name' must be provided")
        result = EvaluationResult(
            name=name,
            score_value=score_value,
            score_label=score_label,
            explanation=explanation,
            response_id=response_id,
            error_type=error_type,
        )
    EvaluationEventEmitter.emit(span, result)


def set_test_suite_context(
    span: Span | None,
    *,
    run_id: str | None = None,
    name: str | None = None,
    status: str | None = None,
) -> None:
    """Set test suite attributes on a span.

    Sets span-level attributes in the ``test.suite.*`` namespace to organize
    evaluations into test suite runs. Only non-None values are set.

    Args:
        span: Target OTel span. Skipped if None or not recording.
        run_id: Unique test suite run identifier (test.suite.run.id).
        name: Human-readable suite name (test.suite.name).
        status: Run status: success, failure, skipped, aborted, timed_out, in_progress
            (test.suite.run.status).
    """
    if span is None or not span.is_recording():
        return
    if run_id is not None:
        span.set_attribute("test.suite.run.id", run_id)
    if name is not None:
        span.set_attribute("test.suite.name", name)
    if status is not None:
        span.set_attribute("test.suite.run.status", status)


def set_test_case_context(
    span: Span | None,
    *,
    case_id: str | None = None,
    name: str | None = None,
    status: str | None = None,
) -> None:
    """Set test case attributes on a span.

    Sets span-level attributes in the ``test.case.*`` namespace to associate
    evaluation spans with specific test cases. Only non-None values are set.

    Args:
        span: Target OTel span. Skipped if None or not recording.
        case_id: Unique test case identifier (test.case.id).
        name: Human-readable case name (test.case.name).
        status: Result status: pass, fail (test.case.result.status).
    """
    if span is None or not span.is_recording():
        return
    if case_id is not None:
        span.set_attribute("test.case.id", case_id)
    if name is not None:
        span.set_attribute("test.case.name", name)
    if status is not None:
        span.set_attribute("test.case.result.status", status)
