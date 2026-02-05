"""GenAI evaluation support for OpenTelemetry integration.

This module provides evaluation capabilities following the OpenTelemetry GenAI Semantic Conventions
for capturing evaluation results as events attached to GenAI operation spans.

Reference: https://github.com/open-telemetry/semantic-conventions/pull/2563
"""

import logging
from typing import Any, Optional

from opentelemetry.trace import Span

from .tracer import get_tracer

logger = logging.getLogger(__name__)


class EvaluationResult:
    """Represents an evaluation result following OpenTelemetry GenAI Semantic Conventions.
    
    Attributes:
        name: Name of the evaluation metric (e.g., "relevance", "hallucination", "accuracy")
        score: Numeric score from the evaluator
        score_label: Human-readable interpretation of the score (optional)
        reasoning: Explanation from the evaluator (optional)
        response_id: Links eval to the completion being evaluated when span linking isn't possible (optional)
    """
    
    def __init__(
        self,
        name: str,
        score: float,
        score_label: str | None = None,
        reasoning: str | None = None,
        response_id: str | None = None,
    ) -> None:
        """Initialize an evaluation result.
        
        Args:
            name: Name of the evaluation metric (e.g., "relevance", "hallucination")
            score: Numeric score from the evaluator
            score_label: Human-readable interpretation of the score
            reasoning: Explanation from the evaluator
            response_id: Links eval to the completion being evaluated
        """
        self.name = name
        self.score = score
        self.score_label = score_label
        self.reasoning = reasoning
        self.response_id = response_id
    
    def to_attributes(self) -> dict[str, Any]:
        """Convert evaluation result to OpenTelemetry event attributes.
        
        Returns:
            Dictionary of attributes following GenAI semantic conventions
        """
        attributes = {
            "gen_ai.evaluation.name": self.name,
            "gen_ai.evaluation.score": self.score,
        }
        
        if self.score_label is not None:
            attributes["gen_ai.evaluation.score.label"] = self.score_label
            
        if self.reasoning is not None:
            attributes["gen_ai.evaluation.reasoning"] = self.reasoning
            
        if self.response_id is not None:
            attributes["gen_ai.response.id"] = self.response_id
            
        return attributes


class EvaluationTracer:
    """Handles GenAI evaluation tracing using OpenTelemetry events."""
    
    def __init__(self) -> None:
        """Initialize the evaluation tracer."""
        self.tracer = get_tracer()
    
    def add_evaluation_event(
        self,
        span: Span,
        evaluation_result: EvaluationResult,
    ) -> None:
        """Add an evaluation event to a span.
        
        Args:
            span: The span to add the evaluation event to (typically the GenAI operation span)
            evaluation_result: The evaluation result to record
        """
        if not span or not span.is_recording():
            logger.debug("span=<%s> | skipping evaluation event for non-recording span", span)
            return
            
        try:
            span.add_event(
                "gen_ai.evaluation.result",
                attributes=evaluation_result.to_attributes()
            )
            logger.debug(
                "evaluation_name=<%s>, score=<%s> | added evaluation event to span",
                evaluation_result.name,
                evaluation_result.score
            )
        except Exception as e:
            logger.warning(
                "evaluation_name=<%s>, error=<%s> | failed to add evaluation event",
                evaluation_result.name,
                e,
                exc_info=True
            )
    
    def add_multiple_evaluation_events(
        self,
        span: Span,
        evaluation_results: list[EvaluationResult],
    ) -> None:
        """Add multiple evaluation events to a span.
        
        Args:
            span: The span to add the evaluation events to
            evaluation_results: List of evaluation results to record
        """
        for result in evaluation_results:
            self.add_evaluation_event(span, result)
    
    def evaluate_and_trace(
        self,
        span: Span,
        evaluator_func: callable,
        content: str,
        evaluation_name: str,
        **evaluator_kwargs: Any,
    ) -> EvaluationResult | None:
        """Run an evaluator function and trace the result.
        
        Args:
            span: The span to add the evaluation event to
            evaluator_func: Function that performs the evaluation
            content: Content to evaluate (e.g., model response)
            evaluation_name: Name of the evaluation metric
            **evaluator_kwargs: Additional arguments to pass to the evaluator
            
        Returns:
            The evaluation result, or None if evaluation failed
        """
        try:
            # Run the evaluator
            result = evaluator_func(content, **evaluator_kwargs)
            
            # Handle different return formats from evaluators
            if isinstance(result, dict):
                evaluation_result = EvaluationResult(
                    name=evaluation_name,
                    score=result.get("score", 0.0),
                    score_label=result.get("label"),
                    reasoning=result.get("reasoning"),
                )
            elif isinstance(result, (int, float)):
                evaluation_result = EvaluationResult(
                    name=evaluation_name,
                    score=float(result),
                )
            else:
                logger.warning(
                    "evaluation_name=<%s>, result_type=<%s> | unsupported evaluator result type",
                    evaluation_name,
                    type(result)
                )
                return None
            
            # Add the evaluation event to the span
            self.add_evaluation_event(span, evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logger.warning(
                "evaluation_name=<%s>, error=<%s> | evaluation failed",
                evaluation_name,
                e,
                exc_info=True
            )
            return None


# Singleton instance for global access
_evaluation_tracer_instance: Optional[EvaluationTracer] = None


def get_evaluation_tracer() -> EvaluationTracer:
    """Get or create the global evaluation tracer.
    
    Returns:
        The global evaluation tracer instance
    """
    global _evaluation_tracer_instance
    
    if not _evaluation_tracer_instance:
        _evaluation_tracer_instance = EvaluationTracer()
    
    return _evaluation_tracer_instance


# Convenience functions for common evaluation patterns
def add_relevance_evaluation(
    span: Span,
    score: float,
    label: str | None = None,
    reasoning: str | None = None,
) -> None:
    """Add a relevance evaluation event to a span.
    
    Args:
        span: The span to add the evaluation to
        score: Relevance score (typically 0.0 to 1.0)
        label: Human-readable label (e.g., "relevant", "not_relevant")
        reasoning: Explanation of the relevance assessment
    """
    evaluation = EvaluationResult(
        name="relevance",
        score=score,
        score_label=label,
        reasoning=reasoning,
    )
    get_evaluation_tracer().add_evaluation_event(span, evaluation)


def add_hallucination_evaluation(
    span: Span,
    score: float,
    label: str | None = None,
    reasoning: str | None = None,
) -> None:
    """Add a hallucination evaluation event to a span.
    
    Args:
        span: The span to add the evaluation to
        score: Hallucination score (typically 0.0 = no hallucination, 1.0 = high hallucination)
        label: Human-readable label (e.g., "factual", "hallucinated")
        reasoning: Explanation of the hallucination assessment
    """
    evaluation = EvaluationResult(
        name="hallucination",
        score=score,
        score_label=label,
        reasoning=reasoning,
    )
    get_evaluation_tracer().add_evaluation_event(span, evaluation)


def add_accuracy_evaluation(
    span: Span,
    score: float,
    label: str | None = None,
    reasoning: str | None = None,
) -> None:
    """Add an accuracy evaluation event to a span.
    
    Args:
        span: The span to add the evaluation to
        score: Accuracy score (typically 0.0 to 1.0)
        label: Human-readable label (e.g., "accurate", "inaccurate")
        reasoning: Explanation of the accuracy assessment
    """
    evaluation = EvaluationResult(
        name="accuracy",
        score=score,
        score_label=label,
        reasoning=reasoning,
    )
    get_evaluation_tracer().add_evaluation_event(span, evaluation)