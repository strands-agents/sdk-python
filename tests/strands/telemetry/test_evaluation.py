"""Tests for GenAI evaluation telemetry functionality."""

from unittest.mock import Mock

from strands.telemetry.evaluation import (
    EvaluationResult,
    EvaluationTracer,
    get_evaluation_tracer,
)


class TestEvaluationResult:
    """Test EvaluationResult class."""

    def test_basic_evaluation_result(self):
        """Test creating a basic evaluation result."""
        result = EvaluationResult(name="relevance", score=0.85)

        assert result.name == "relevance"
        assert result.score == 0.85
        assert result.score_label is None
        assert result.reasoning is None
        assert result.response_id is None

    def test_full_evaluation_result(self):
        """Test creating a full evaluation result with all fields."""
        result = EvaluationResult(
            name="hallucination",
            score=0.1,
            score_label="factual",
            reasoning="Response contains verifiable facts",
            response_id="resp_123",
        )

        assert result.name == "hallucination"
        assert result.score == 0.1
        assert result.score_label == "factual"
        assert result.reasoning == "Response contains verifiable facts"
        assert result.response_id == "resp_123"

    def test_to_attributes_basic(self):
        """Test converting basic evaluation result to attributes."""
        result = EvaluationResult(name="accuracy", score=0.9)
        attributes = result.to_attributes()

        expected = {
            "gen_ai.evaluation.name": "accuracy",
            "gen_ai.evaluation.score": 0.9,
        }
        assert attributes == expected

    def test_to_attributes_full(self):
        """Test converting full evaluation result to attributes."""
        result = EvaluationResult(
            name="relevance",
            score=0.75,
            score_label="relevant",
            reasoning="Query terms found in response",
            response_id="resp_456",
        )
        attributes = result.to_attributes()

        expected = {
            "gen_ai.evaluation.name": "relevance",
            "gen_ai.evaluation.score": 0.75,
            "gen_ai.evaluation.score.label": "relevant",
            "gen_ai.evaluation.reasoning": "Query terms found in response",
            "gen_ai.response.id": "resp_456",
        }
        assert attributes == expected


class TestEvaluationTracer:
    """Test EvaluationTracer class."""

    def test_add_evaluation_event(self):
        """Test adding an evaluation event to a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True

        tracer = EvaluationTracer()
        result = EvaluationResult(name="test_metric", score=0.8, score_label="good")

        tracer.add_evaluation_event(mock_span, result)

        mock_span.add_event.assert_called_once_with(
            "gen_ai.evaluation.result",
            attributes={
                "gen_ai.evaluation.name": "test_metric",
                "gen_ai.evaluation.score": 0.8,
                "gen_ai.evaluation.score.label": "good",
            },
        )

    def test_add_evaluation_event_non_recording_span(self):
        """Test that evaluation events are skipped for non-recording spans."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False

        tracer = EvaluationTracer()
        result = EvaluationResult(name="test", score=0.5)

        tracer.add_evaluation_event(mock_span, result)

        mock_span.add_event.assert_not_called()

    def test_add_evaluation_event_none_span(self):
        """Test that evaluation events handle None spans gracefully."""
        tracer = EvaluationTracer()
        result = EvaluationResult(name="test", score=0.5)

        # Should not raise an exception
        tracer.add_evaluation_event(None, result)


class TestSingleton:
    """Test singleton behavior of get_evaluation_tracer."""

    def test_get_evaluation_tracer_singleton(self):
        """Test that get_evaluation_tracer returns the same instance."""
        tracer1 = get_evaluation_tracer()
        tracer2 = get_evaluation_tracer()

        assert tracer1 is tracer2
        assert isinstance(tracer1, EvaluationTracer)
