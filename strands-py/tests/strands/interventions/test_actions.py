"""Tests for intervention action types and factory functions."""

import dataclasses

import pytest

from strands.interventions.actions import (
    confirm,
    default_evaluate,
    deny,
    guide,
    proceed,
    transform,
)


class TestDefaultEvaluate:
    @pytest.mark.parametrize(
        "response,expected",
        [
            (True, True),
            ("yes", True),
            ("y", True),
            ("Y", True),
            ("YES", True),
            ("  yes  ", True),
            ("no", False),
            (False, False),
            (None, False),
            ("", False),
            (0, False),
            (1, False),
        ],
    )
    def test_responses(self, response, expected):
        assert default_evaluate(response) is expected


class TestFactoryFunctions:
    def test_proceed(self):
        action = proceed()
        assert action.type == "proceed"
        assert action.reason is None

    def test_proceed_with_reason(self):
        action = proceed(reason="all good")
        assert action.reason == "all good"

    def test_deny(self):
        action = deny("not authorized")
        assert action.type == "deny"
        assert action.reason == "not authorized"

    def test_guide(self):
        action = guide("try a different approach")
        assert action.type == "guide"
        assert action.feedback == "try a different approach"
        assert action.reason is None

    def test_guide_with_reason(self):
        action = guide("try again", reason="too vague")
        assert action.reason == "too vague"

    def test_confirm(self):
        action = confirm("approve?")
        assert action.type == "confirm"
        assert action.prompt == "approve?"
        assert action.response is None
        assert action.evaluate is default_evaluate

    def test_confirm_with_response(self):
        action = confirm("approve?", response="yes")
        assert action.response == "yes"

    def test_confirm_with_custom_evaluate(self):
        def custom(r):
            return r == "magic"

        action = confirm("approve?", evaluate=custom)
        assert action.evaluate is custom

    def test_transform(self):
        def fn(e):
            return None

        action = transform(fn)
        assert action.type == "transform"
        assert action.apply is fn
        assert action.reason is None

    def test_transform_with_reason(self):
        def noop(e):
            return None

        action = transform(noop, reason="redacted PII")
        assert action.reason == "redacted PII"


class TestDataclassImmutability:
    def test_proceed_is_frozen(self):
        action = proceed()
        with pytest.raises(dataclasses.FrozenInstanceError):
            action.reason = "changed"

    def test_deny_is_frozen(self):
        action = deny("reason")
        with pytest.raises(dataclasses.FrozenInstanceError):
            action.reason = "changed"
