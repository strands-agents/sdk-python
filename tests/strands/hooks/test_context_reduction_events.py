"""Tests for BeforeContextReductionEvent and AfterContextReductionEvent."""

from unittest.mock import Mock

import pytest

from strands.hooks import AfterContextReductionEvent, BeforeContextReductionEvent
from strands.types.exceptions import ContextWindowOverflowException


@pytest.fixture
def agent():
    return Mock()


@pytest.fixture
def context_overflow_exception():
    return ContextWindowOverflowException("Context window exceeded")


@pytest.fixture
def before_context_reduction_event(agent, context_overflow_exception):
    return BeforeContextReductionEvent(
        agent=agent,
        exception=context_overflow_exception,
        message_count=100,
    )


@pytest.fixture
def after_context_reduction_event(agent):
    return AfterContextReductionEvent(
        agent=agent,
        original_message_count=100,
        new_message_count=50,
        removed_count=50,
    )


class TestBeforeContextReductionEvent:
    """Tests for BeforeContextReductionEvent."""

    def test_event_has_correct_attributes(self, before_context_reduction_event, agent, context_overflow_exception):
        """Test that BeforeContextReductionEvent has correct attribute values."""
        assert before_context_reduction_event.agent is agent
        assert before_context_reduction_event.exception is context_overflow_exception
        assert before_context_reduction_event.message_count == 100

    def test_event_should_not_reverse_callbacks(self, before_context_reduction_event):
        """Test that BeforeContextReductionEvent does not reverse callbacks (default behavior)."""
        assert before_context_reduction_event.should_reverse_callbacks == False  # noqa: E712

    def test_agent_not_writable(self, before_context_reduction_event):
        """Test that agent property is not writable."""
        with pytest.raises(AttributeError, match="Property agent is not writable"):
            before_context_reduction_event.agent = Mock()

    def test_exception_not_writable(self, before_context_reduction_event):
        """Test that exception property is not writable."""
        with pytest.raises(AttributeError, match="Property exception is not writable"):
            before_context_reduction_event.exception = ContextWindowOverflowException("new")

    def test_message_count_not_writable(self, before_context_reduction_event):
        """Test that message_count property is not writable."""
        with pytest.raises(AttributeError, match="Property message_count is not writable"):
            before_context_reduction_event.message_count = 50


class TestAfterContextReductionEvent:
    """Tests for AfterContextReductionEvent."""

    def test_event_has_correct_attributes(self, after_context_reduction_event, agent):
        """Test that AfterContextReductionEvent has correct attribute values."""
        assert after_context_reduction_event.agent is agent
        assert after_context_reduction_event.original_message_count == 100
        assert after_context_reduction_event.new_message_count == 50
        assert after_context_reduction_event.removed_count == 50

    def test_event_should_reverse_callbacks(self, after_context_reduction_event):
        """Test that AfterContextReductionEvent reverses callbacks (cleanup behavior)."""
        assert after_context_reduction_event.should_reverse_callbacks == True  # noqa: E712

    def test_agent_not_writable(self, after_context_reduction_event):
        """Test that agent property is not writable."""
        with pytest.raises(AttributeError, match="Property agent is not writable"):
            after_context_reduction_event.agent = Mock()

    def test_original_message_count_not_writable(self, after_context_reduction_event):
        """Test that original_message_count property is not writable."""
        with pytest.raises(AttributeError, match="Property original_message_count is not writable"):
            after_context_reduction_event.original_message_count = 200

    def test_new_message_count_not_writable(self, after_context_reduction_event):
        """Test that new_message_count property is not writable."""
        with pytest.raises(AttributeError, match="Property new_message_count is not writable"):
            after_context_reduction_event.new_message_count = 25

    def test_removed_count_not_writable(self, after_context_reduction_event):
        """Test that removed_count property is not writable."""
        with pytest.raises(AttributeError, match="Property removed_count is not writable"):
            after_context_reduction_event.removed_count = 75

    def test_removed_count_calculation(self, agent):
        """Test that removed_count is correctly calculated."""
        event = AfterContextReductionEvent(
            agent=agent,
            original_message_count=150,
            new_message_count=80,
            removed_count=150 - 80,
        )
        assert event.removed_count == 70
        assert event.original_message_count - event.new_message_count == event.removed_count
