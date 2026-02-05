"""Tests for GenAI evaluation telemetry functionality."""

import pytest
from unittest.mock import Mock, patch

from strands.telemetry.evaluation import (
    EvaluationResult,
    EvaluationTracer,
    add_accuracy_evaluation,
    add_hallucination_evaluation,
    add_relevance_evaluation,
    get_evaluation_tracer,
)


class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_basic_evaluation_result(self):
        """Test creating a basic evaluation result."""
        result = EvaluationResult(
            name="relevance",
            score=0.85
        )
        
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
            response_id="resp_123"
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
            response_id="resp_456"
        )
        attributes = result.to_attributes()
        
        expected = {
            "gen_ai.evaluation.name": "relevance",
            "gen_ai.evaluation.score": 0.75,
            "gen_ai.evaluation.score.label": "relevant",
            "gen_ai.evaluation.reasoning": "Query terms found in response",
            "gen_ai.response.id": "resp_456"
        }
        assert attributes == expected


class TestEvaluationTracer:
    """Test EvaluationTracer class."""
    
    def test_add_evaluation_event(self):
        """Test adding an evaluation event to a span."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        tracer = EvaluationTracer()
        result = EvaluationResult(
            name="test_metric",
            score=0.8,
            score_label="good"
        )
        
        tracer.add_evaluation_event(mock_span, result)
        
        mock_span.add_event.assert_called_once_with(
            "gen_ai.evaluation.result",
            attributes={
                "gen_ai.evaluation.name": "test_metric",
                "gen_ai.evaluation.score": 0.8,
                "gen_ai.evaluation.score.label": "good"
            }
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
    
    def test_add_multiple_evaluation_events(self):
        """Test adding multiple evaluation events."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        tracer = EvaluationTracer()
        results = [
            EvaluationResult(name="metric1", score=0.7),
            EvaluationResult(name="metric2", score=0.9),
        ]
        
        tracer.add_multiple_evaluation_events(mock_span, results)
        
        assert mock_span.add_event.call_count == 2
        
        # Check first call
        first_call = mock_span.add_event.call_args_list[0]
        assert first_call[0][0] == "gen_ai.evaluation.result"
        assert first_call[1]["attributes"]["gen_ai.evaluation.name"] == "metric1"
        
        # Check second call
        second_call = mock_span.add_event.call_args_list[1]
        assert second_call[0][0] == "gen_ai.evaluation.result"
        assert second_call[1]["attributes"]["gen_ai.evaluation.name"] == "metric2"
    
    def test_evaluate_and_trace_dict_result(self):
        """Test evaluate_and_trace with dict result from evaluator."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        def mock_evaluator(content, **kwargs):
            return {
                "score": 0.85,
                "label": "good",
                "reasoning": "Test reasoning"
            }
        
        tracer = EvaluationTracer()
        result = tracer.evaluate_and_trace(
            mock_span,
            mock_evaluator,
            "test content",
            "test_metric"
        )
        
        assert result is not None
        assert result.name == "test_metric"
        assert result.score == 0.85
        assert result.score_label == "good"
        assert result.reasoning == "Test reasoning"
        
        mock_span.add_event.assert_called_once()
    
    def test_evaluate_and_trace_numeric_result(self):
        """Test evaluate_and_trace with numeric result from evaluator."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        def mock_evaluator(content, **kwargs):
            return 0.75
        
        tracer = EvaluationTracer()
        result = tracer.evaluate_and_trace(
            mock_span,
            mock_evaluator,
            "test content",
            "test_metric"
        )
        
        assert result is not None
        assert result.name == "test_metric"
        assert result.score == 0.75
        assert result.score_label is None
        
        mock_span.add_event.assert_called_once()
    
    def test_evaluate_and_trace_invalid_result(self):
        """Test evaluate_and_trace with invalid result from evaluator."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        def mock_evaluator(content, **kwargs):
            return "invalid result"
        
        tracer = EvaluationTracer()
        result = tracer.evaluate_and_trace(
            mock_span,
            mock_evaluator,
            "test content",
            "test_metric"
        )
        
        assert result is None
        mock_span.add_event.assert_not_called()
    
    def test_evaluate_and_trace_exception(self):
        """Test evaluate_and_trace when evaluator raises exception."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        
        def mock_evaluator(content, **kwargs):
            raise ValueError("Evaluator failed")
        
        tracer = EvaluationTracer()
        result = tracer.evaluate_and_trace(
            mock_span,
            mock_evaluator,
            "test content",
            "test_metric"
        )
        
        assert result is None
        mock_span.add_event.assert_not_called()


class TestConvenienceFunctions:
    """Test convenience functions for common evaluation types."""
    
    @patch('strands.telemetry.evaluation.get_evaluation_tracer')
    def test_add_relevance_evaluation(self, mock_get_tracer):
        """Test add_relevance_evaluation convenience function."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = Mock()
        
        add_relevance_evaluation(
            mock_span,
            score=0.9,
            label="relevant",
            reasoning="Test reasoning"
        )
        
        mock_tracer.add_evaluation_event.assert_called_once()
        call_args = mock_tracer.add_evaluation_event.call_args
        
        assert call_args[0][0] == mock_span  # span argument
        result = call_args[0][1]  # evaluation result argument
        assert result.name == "relevance"
        assert result.score == 0.9
        assert result.score_label == "relevant"
        assert result.reasoning == "Test reasoning"
    
    @patch('strands.telemetry.evaluation.get_evaluation_tracer')
    def test_add_hallucination_evaluation(self, mock_get_tracer):
        """Test add_hallucination_evaluation convenience function."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = Mock()
        
        add_hallucination_evaluation(
            mock_span,
            score=0.1,
            label="factual"
        )
        
        mock_tracer.add_evaluation_event.assert_called_once()
        call_args = mock_tracer.add_evaluation_event.call_args
        
        result = call_args[0][1]
        assert result.name == "hallucination"
        assert result.score == 0.1
        assert result.score_label == "factual"
    
    @patch('strands.telemetry.evaluation.get_evaluation_tracer')
    def test_add_accuracy_evaluation(self, mock_get_tracer):
        """Test add_accuracy_evaluation convenience function."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = Mock()
        
        add_accuracy_evaluation(
            mock_span,
            score=0.95,
            reasoning="Highly accurate response"
        )
        
        mock_tracer.add_evaluation_event.assert_called_once()
        call_args = mock_tracer.add_evaluation_event.call_args
        
        result = call_args[0][1]
        assert result.name == "accuracy"
        assert result.score == 0.95
        assert result.reasoning == "Highly accurate response"


class TestSingleton:
    """Test singleton behavior of get_evaluation_tracer."""
    
    def test_get_evaluation_tracer_singleton(self):
        """Test that get_evaluation_tracer returns the same instance."""
        tracer1 = get_evaluation_tracer()
        tracer2 = get_evaluation_tracer()
        
        assert tracer1 is tracer2
        assert isinstance(tracer1, EvaluationTracer)