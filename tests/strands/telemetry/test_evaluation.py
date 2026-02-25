"""Tests for GenAI evaluation result telemetry."""

from unittest.mock import MagicMock, call

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from opentelemetry.trace import Span

import strands.telemetry
from strands.telemetry.evaluation import (
    EvaluationEventEmitter,
    EvaluationResult,
    add_evaluation_event,
    set_test_case_context,
    set_test_suite_context,
)

# Hypothesis strategy for generating arbitrary EvaluationResult instances
evaluation_results = st.builds(
    EvaluationResult,
    name=st.text(min_size=1, max_size=50),
    score_value=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    score_label=st.one_of(st.none(), st.text(max_size=50)),
    explanation=st.one_of(st.none(), st.text(max_size=200)),
    response_id=st.one_of(st.none(), st.text(max_size=50)),
    error_type=st.one_of(st.none(), st.text(max_size=50)),
)

# Field-to-OTel-key mapping for optional fields
_OPTIONAL_FIELD_TO_OTEL_KEY = {
    "score_value": "gen_ai.evaluation.score.value",
    "score_label": "gen_ai.evaluation.score.label",
    "explanation": "gen_ai.evaluation.explanation",
    "response_id": "gen_ai.response.id",
    "error_type": "error.type",
}

# Parameter-to-OTel-key mapping for test suite context
_SUITE_PARAM_TO_OTEL_KEY = {
    "run_id": "test.suite.run.id",
    "name": "test.suite.name",
    "status": "test.suite.run.status",
}

# Well-known status values for test suite runs
_SUITE_STATUS_VALUES = ["success", "failure", "skipped", "aborted", "timed_out", "in_progress"]

# Parameter-to-OTel-key mapping for test case context
_CASE_PARAM_TO_OTEL_KEY = {
    "case_id": "test.case.id",
    "name": "test.case.name",
    "status": "test.case.result.status",
}


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""

    def test_required_name_field(self):
        """Test that name is required and stored correctly."""
        result = EvaluationResult(name="accuracy")
        assert result.name == "accuracy"

    def test_optional_fields_default_to_none(self):
        """Test that all optional fields default to None."""
        result = EvaluationResult(name="accuracy")
        assert result.score_value is None
        assert result.score_label is None
        assert result.explanation is None
        assert result.response_id is None
        assert result.error_type is None

    def test_all_fields_set(self):
        """Test constructing with all fields populated."""
        result = EvaluationResult(
            name="accuracy",
            score_value=0.95,
            score_label="pass",
            explanation="High accuracy on test set",
            response_id="chatcmpl-abc123",
            error_type="TimeoutError",
        )
        assert result.name == "accuracy"
        assert result.score_value == 0.95
        assert result.score_label == "pass"
        assert result.explanation == "High accuracy on test set"
        assert result.response_id == "chatcmpl-abc123"
        assert result.error_type == "TimeoutError"


class TestToOtelAttributes:
    """Tests for EvaluationResult.to_otel_attributes()."""

    def test_name_only(self):
        """Test attributes with only the required name field."""
        result = EvaluationResult(name="accuracy")
        attrs = result.to_otel_attributes()
        assert attrs == {"gen_ai.evaluation.name": "accuracy"}

    def test_all_fields_populated(self):
        """Test attributes with all fields set."""
        result = EvaluationResult(
            name="accuracy",
            score_value=0.95,
            score_label="pass",
            explanation="Good result",
            response_id="resp-123",
            error_type="ValueError",
        )
        attrs = result.to_otel_attributes()
        assert attrs == {
            "gen_ai.evaluation.name": "accuracy",
            "gen_ai.evaluation.score.value": 0.95,
            "gen_ai.evaluation.score.label": "pass",
            "gen_ai.evaluation.explanation": "Good result",
            "gen_ai.response.id": "resp-123",
            "error.type": "ValueError",
        }

    def test_omits_none_fields(self):
        """Test that None fields are omitted from the output."""
        result = EvaluationResult(
            name="tone",
            score_value=0.88,
            score_label=None,
            explanation=None,
            response_id="resp-456",
            error_type=None,
        )
        attrs = result.to_otel_attributes()
        assert attrs == {
            "gen_ai.evaluation.name": "tone",
            "gen_ai.evaluation.score.value": 0.88,
            "gen_ai.response.id": "resp-456",
        }
        assert "gen_ai.evaluation.score.label" not in attrs
        assert "gen_ai.evaluation.explanation" not in attrs
        assert "error.type" not in attrs

    def test_score_value_zero_is_included(self):
        """Test that a score_value of 0.0 is included (not treated as falsy)."""
        result = EvaluationResult(name="accuracy", score_value=0.0)
        attrs = result.to_otel_attributes()
        assert attrs == {
            "gen_ai.evaluation.name": "accuracy",
            "gen_ai.evaluation.score.value": 0.0,
        }

    def test_empty_string_fields_included(self):
        """Test that empty strings are included (not treated as None)."""
        result = EvaluationResult(name="accuracy", score_label="", explanation="")
        attrs = result.to_otel_attributes()
        assert attrs == {
            "gen_ai.evaluation.name": "accuracy",
            "gen_ai.evaluation.score.label": "",
            "gen_ai.evaluation.explanation": "",
        }

    def test_name_always_present(self):
        """Test that gen_ai.evaluation.name is always in the output."""
        result = EvaluationResult(name="test-metric")
        attrs = result.to_otel_attributes()
        assert "gen_ai.evaluation.name" in attrs
        assert attrs["gen_ai.evaluation.name"] == "test-metric"

    def test_correct_otel_attribute_keys(self):
        """Test that all OTel attribute keys match the semantic conventions."""
        result = EvaluationResult(
            name="accuracy",
            score_value=1.0,
            score_label="perfect",
            explanation="Exact match",
            response_id="resp-789",
            error_type="None",
        )
        attrs = result.to_otel_attributes()
        expected_keys = {
            "gen_ai.evaluation.name",
            "gen_ai.evaluation.score.value",
            "gen_ai.evaluation.score.label",
            "gen_ai.evaluation.explanation",
            "gen_ai.response.id",
            "error.type",
        }
        assert set(attrs.keys()) == expected_keys


class TestToOtelAttributesProperty:
    """Property-based tests for EvaluationResult.to_otel_attributes().

    **Validates: Requirements 1.7, 1.8**
    """

    @given(result=evaluation_results)
    @settings(max_examples=100)
    def test_name_always_present_in_attributes(self, result: EvaluationResult) -> None:
        """Property 1a: The output always contains gen_ai.evaluation.name mapped to the name field.

        **Validates: Requirements 1.7, 1.8**
        """
        attrs = result.to_otel_attributes()
        assert "gen_ai.evaluation.name" in attrs
        assert attrs["gen_ai.evaluation.name"] == result.name

    @given(result=evaluation_results)
    @settings(max_examples=100)
    def test_non_none_fields_present_with_correct_keys(self, result: EvaluationResult) -> None:
        """Property 1b: Each non-None optional field appears with its correct OTel key.

        **Validates: Requirements 1.7, 1.8**
        """
        attrs = result.to_otel_attributes()
        for field_name, otel_key in _OPTIONAL_FIELD_TO_OTEL_KEY.items():
            field_value = getattr(result, field_name)
            if field_value is not None:
                assert otel_key in attrs, f"Expected key {otel_key!r} for non-None field {field_name!r}"
                assert attrs[otel_key] == field_value

    @given(result=evaluation_results)
    @settings(max_examples=100)
    def test_none_fields_omitted(self, result: EvaluationResult) -> None:
        """Property 1c: Keys for None-valued fields do NOT appear in the output.

        **Validates: Requirements 1.7, 1.8**
        """
        attrs = result.to_otel_attributes()
        for field_name, otel_key in _OPTIONAL_FIELD_TO_OTEL_KEY.items():
            field_value = getattr(result, field_name)
            if field_value is None:
                assert otel_key not in attrs, f"Key {otel_key!r} should be absent for None field {field_name!r}"

    @given(result=evaluation_results)
    @settings(max_examples=100)
    def test_output_contains_exactly_expected_keys(self, result: EvaluationResult) -> None:
        """Property 1d: The output dict contains exactly the expected set of keys — no more, no less.

        **Validates: Requirements 1.7, 1.8**
        """
        attrs = result.to_otel_attributes()

        expected_keys = {"gen_ai.evaluation.name"}
        for field_name, otel_key in _OPTIONAL_FIELD_TO_OTEL_KEY.items():
            if getattr(result, field_name) is not None:
                expected_keys.add(otel_key)

        assert set(attrs.keys()) == expected_keys


class TestEmitEventProperty:
    """Property-based tests for EvaluationEventEmitter.emit().

    **Validates: Requirements 2.1, 2.2**
    """

    @given(result=evaluation_results)
    @settings(max_examples=100)
    def test_emit_calls_add_event_exactly_once(self, result: EvaluationResult) -> None:
        """Property 2: emit() invokes span.add_event() exactly once with correct name and attributes.

        For any EvaluationResult and any recording Span, calling EvaluationEventEmitter.emit(span,
        result) SHALL invoke span.add_event() exactly once with event name
        "gen_ai.evaluation.result" and attributes equal to result.to_otel_attributes().

        **Validates: Requirements 2.1, 2.2**
        """
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        EvaluationEventEmitter.emit(span, result)

        span.add_event.assert_called_once_with(
            "gen_ai.evaluation.result",
            attributes=result.to_otel_attributes(),
        )


class TestMultipleEmissionProperty:
    """Property-based tests for multiple emission count invariant.

    **Validates: Requirements 2.4**
    """

    @given(results=st.lists(evaluation_results, min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_emission_count_equals_result_count(self, results: list[EvaluationResult]) -> None:
        """Property 3: Multiple emission count invariant.

        For any list of EvaluationResult instances of length N and any recording Span, emitting
        all N results on that span SHALL result in exactly N calls to span.add_event(), each with
        event name "gen_ai.evaluation.result".

        **Validates: Requirements 2.4**
        """
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        for result in results:
            EvaluationEventEmitter.emit(span, result)

        assert span.add_event.call_count == len(results)

        for recorded_call in span.add_event.call_args_list:
            assert recorded_call[0][0] == "gen_ai.evaluation.result"


class TestConvenienceFunctionProperty:
    """Property-based tests for add_evaluation_event convenience function equivalence.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        name=st.text(min_size=1, max_size=50),
        score_value=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
        score_label=st.one_of(st.none(), st.text(max_size=50)),
        explanation=st.one_of(st.none(), st.text(max_size=200)),
        response_id=st.one_of(st.none(), st.text(max_size=50)),
        error_type=st.one_of(st.none(), st.text(max_size=50)),
    )
    @settings(max_examples=100)
    def test_convenience_function_matches_explicit_emit(
        self,
        name: str,
        score_value: float | None,
        score_label: str | None,
        explanation: str | None,
        response_id: str | None,
        error_type: str | None,
    ) -> None:
        """Property 4: Convenience function equivalence.

        For any set of keyword arguments (name, score_value, score_label, explanation,
        response_id, error_type) and any recording Span, calling add_evaluation_event(span,
        name=name, ...) SHALL produce the same span.add_event() call as constructing
        EvaluationResult(name=name, ...) and calling EvaluationEventEmitter.emit(span, result).

        **Validates: Requirements 3.1, 3.2**
        """
        span1 = MagicMock(spec=Span)
        span1.is_recording.return_value = True
        span2 = MagicMock(spec=Span)
        span2.is_recording.return_value = True

        add_evaluation_event(
            span1,
            name=name,
            score_value=score_value,
            score_label=score_label,
            explanation=explanation,
            response_id=response_id,
            error_type=error_type,
        )

        result = EvaluationResult(
            name=name,
            score_value=score_value,
            score_label=score_label,
            explanation=explanation,
            response_id=response_id,
            error_type=error_type,
        )
        EvaluationEventEmitter.emit(span2, result)

        span1.add_event.assert_called_once()
        span2.add_event.assert_called_once()
        assert span1.add_event.call_args == span2.add_event.call_args


class TestEdgeCases:
    """Unit tests for edge cases: None span, non-recording span, missing name ValueError.

    Validates: Requirements 2.3, 3.3
    """

    def test_emit_none_span(self) -> None:
        """EvaluationEventEmitter.emit(None, result) should not raise."""
        result = EvaluationResult(name="accuracy", score_value=0.95)
        EvaluationEventEmitter.emit(None, result)

    def test_emit_non_recording_span(self) -> None:
        """emit() on a non-recording span should skip without calling add_event."""
        span = MagicMock(spec=Span)
        span.is_recording.return_value = False

        result = EvaluationResult(name="accuracy", score_value=0.9)
        EvaluationEventEmitter.emit(span, result)

        span.add_event.assert_not_called()

    def test_convenience_none_span(self) -> None:
        """add_evaluation_event(None, name="test") should not raise."""
        add_evaluation_event(None, name="test")

    def test_convenience_non_recording_span(self) -> None:
        """add_evaluation_event on a non-recording span should skip without calling add_event."""
        span = MagicMock(spec=Span)
        span.is_recording.return_value = False

        add_evaluation_event(span, name="test", score_value=0.5)

        span.add_event.assert_not_called()

    def test_convenience_missing_name_raises(self) -> None:
        """add_evaluation_event with no result and no name should raise ValueError."""
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        with pytest.raises(ValueError, match="Either 'result' or 'name' must be provided"):
            add_evaluation_event(span)

    def test_convenience_with_result_object(self) -> None:
        """add_evaluation_event with a pre-built EvaluationResult should emit correctly."""
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        result = EvaluationResult(name="test", score_value=0.85, score_label="good")
        add_evaluation_event(span, result=result)

        span.add_event.assert_called_once_with(
            "gen_ai.evaluation.result",
            attributes={
                "gen_ai.evaluation.name": "test",
                "gen_ai.evaluation.score.value": 0.85,
                "gen_ai.evaluation.score.label": "good",
            },
        )


class TestSuiteContextProperty:
    """Property-based tests for set_test_suite_context attribute correctness.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.5**
    """

    @given(
        run_id=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        status=st.one_of(st.none(), st.sampled_from(_SUITE_STATUS_VALUES)),
    )
    @settings(max_examples=100)
    def test_set_attribute_called_for_each_non_none_param(
        self,
        run_id: str | None,
        name: str | None,
        status: str | None,
    ) -> None:
        """Property 5: Test suite context attribute correctness.

        For any combination of run_id, name, and status values (each either a string or None)
        and any recording Span, calling set_test_suite_context(span, run_id=run_id, name=name,
        status=status) SHALL call span.set_attribute() exactly once for each non-None parameter
        with the correct OTel attribute key, and SHALL NOT call span.set_attribute() for any
        None parameter.

        **Validates: Requirements 4.1, 4.2, 4.3, 4.5**
        """
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        set_test_suite_context(span, run_id=run_id, name=name, status=status)

        params = {"run_id": run_id, "name": name, "status": status}
        expected_calls = []
        for param_name, param_value in params.items():
            if param_value is not None:
                otel_key = _SUITE_PARAM_TO_OTEL_KEY[param_name]
                expected_calls.append(call(otel_key, param_value))

        assert span.set_attribute.call_count == len(expected_calls), (
            f"Expected {len(expected_calls)} set_attribute calls, got {span.set_attribute.call_count}"
        )

        for expected in expected_calls:
            assert expected in span.set_attribute.call_args_list, (
                f"Expected call {expected} not found in {span.set_attribute.call_args_list}"
            )

        actual_keys = [c[0][0] for c in span.set_attribute.call_args_list]
        for param_name, param_value in params.items():
            if param_value is None:
                otel_key = _SUITE_PARAM_TO_OTEL_KEY[param_name]
                assert otel_key not in actual_keys, (
                    f"OTel key {otel_key!r} should NOT be set for None parameter {param_name!r}"
                )


class TestCaseContextProperty:
    """Property-based tests for set_test_case_context attribute correctness.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.5**
    """

    @given(
        case_id=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        status=st.one_of(st.none(), st.sampled_from(["pass", "fail"])),
    )
    @settings(max_examples=100)
    def test_set_attribute_called_for_each_non_none_param(
        self,
        case_id: str | None,
        name: str | None,
        status: str | None,
    ) -> None:
        """Property 6: Test case context attribute correctness.

        For any combination of case_id, name, and status values (each either a string or None)
        and any recording Span, calling set_test_case_context(span, case_id=case_id, name=name,
        status=status) SHALL call span.set_attribute() exactly once for each non-None parameter
        with the correct OTel attribute key, and SHALL NOT call span.set_attribute() for any
        None parameter.

        **Validates: Requirements 5.1, 5.2, 5.3, 5.5**
        """
        span = MagicMock(spec=Span)
        span.is_recording.return_value = True

        set_test_case_context(span, case_id=case_id, name=name, status=status)

        params = {"case_id": case_id, "name": name, "status": status}
        expected_calls = []
        for param_name, param_value in params.items():
            if param_value is not None:
                otel_key = _CASE_PARAM_TO_OTEL_KEY[param_name]
                expected_calls.append(call(otel_key, param_value))

        assert span.set_attribute.call_count == len(expected_calls), (
            f"Expected {len(expected_calls)} set_attribute calls, got {span.set_attribute.call_count}"
        )

        for expected in expected_calls:
            assert expected in span.set_attribute.call_args_list, (
                f"Expected call {expected} not found in {span.set_attribute.call_args_list}"
            )

        actual_keys = [c[0][0] for c in span.set_attribute.call_args_list]
        for param_name, param_value in params.items():
            if param_value is None:
                otel_key = _CASE_PARAM_TO_OTEL_KEY[param_name]
                assert otel_key not in actual_keys, (
                    f"OTel key {otel_key!r} should NOT be set for None parameter {param_name!r}"
                )


class TestPublicAPIExports:
    """Unit tests verifying all evaluation symbols are importable from strands.telemetry.

    Validates: Requirements 6.1, 6.2
    """

    def test_evaluation_result_importable(self) -> None:
        """EvaluationResult is importable from strands.telemetry."""
        assert hasattr(strands.telemetry, "EvaluationResult")
        from strands.telemetry import EvaluationResult as ER

        assert ER is not None

    def test_evaluation_event_emitter_importable(self) -> None:
        """EvaluationEventEmitter is importable from strands.telemetry."""
        assert hasattr(strands.telemetry, "EvaluationEventEmitter")
        from strands.telemetry import EvaluationEventEmitter as EEE

        assert EEE is not None

    def test_add_evaluation_event_importable(self) -> None:
        """add_evaluation_event is importable from strands.telemetry."""
        assert hasattr(strands.telemetry, "add_evaluation_event")
        from strands.telemetry import add_evaluation_event as aee

        assert aee is not None

    def test_set_test_suite_context_importable(self) -> None:
        """set_test_suite_context is importable from strands.telemetry."""
        assert hasattr(strands.telemetry, "set_test_suite_context")
        from strands.telemetry import set_test_suite_context as stsc

        assert stsc is not None

    def test_set_test_case_context_importable(self) -> None:
        """set_test_case_context is importable from strands.telemetry."""
        assert hasattr(strands.telemetry, "set_test_case_context")
        from strands.telemetry import set_test_case_context as stcc

        assert stcc is not None

    def test_all_evaluation_symbols_in_dunder_all(self) -> None:
        """All 5 evaluation symbols are listed in strands.telemetry.__all__."""
        expected_symbols = {
            "EvaluationResult",
            "EvaluationEventEmitter",
            "add_evaluation_event",
            "set_test_suite_context",
            "set_test_case_context",
        }
        module_all = set(strands.telemetry.__all__)
        assert expected_symbols.issubset(module_all), f"Missing from __all__: {expected_symbols - module_all}"
