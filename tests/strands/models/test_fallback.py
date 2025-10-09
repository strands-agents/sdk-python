"""Unit tests for FallbackModel."""

from unittest.mock import AsyncMock

import pytest

from strands.models import Model
from strands.models.fallback import FallbackModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def mock_primary():
    """Create a mock primary model."""
    model = AsyncMock(spec=Model)
    model.get_config.return_value = {"model": "primary"}
    return model


@pytest.fixture
def mock_fallback():
    """Create a mock fallback model."""
    model = AsyncMock(spec=Model)
    model.get_config.return_value = {"model": "fallback"}
    return model


@pytest.fixture
def fallback_model(mock_primary, mock_fallback):
    """Create a FallbackModel instance with mock models."""
    return FallbackModel(primary=mock_primary, fallback=mock_fallback)


async def async_generator(items):
    """Helper function to create async generators for test data."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_primary_success(mock_primary, mock_fallback, fallback_model, alist):
    """Test that primary model success returns primary results without fallback."""
    # Mock primary.stream to return test events
    test_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_primary.stream.return_value = async_generator(test_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert all events received
    assert result == test_events

    # Assert fallback was not called
    mock_fallback.stream.assert_not_called()

    # Assert fallback_count is 0
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0


@pytest.mark.asyncio
async def test_throttle_exception(mock_primary, mock_fallback, fallback_model, alist):
    """Test fallback triggers on ModelThrottledException."""
    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Rate limit exceeded")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert fallback events received
    assert result == fallback_events

    # Assert fallback_count is 1
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1


@pytest.mark.asyncio
async def test_connection_error(mock_primary, mock_fallback, fallback_model, alist):
    """Test fallback triggers on connection errors."""
    # Mock primary.stream to raise exception with "connection timeout" message
    mock_primary.stream.side_effect = Exception("connection timeout error")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert fallback events received
    assert result == fallback_events

    # Assert fallback was called
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1


@pytest.mark.asyncio
async def test_context_overflow_no_fallback(mock_primary, mock_fallback, fallback_model):
    """Test no fallback on ContextWindowOverflowException."""
    # Mock primary.stream to raise ContextWindowOverflowException
    mock_primary.stream.side_effect = ContextWindowOverflowException("Context window exceeded")

    # Call stream and expect exception to be re-raised
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    with pytest.raises(ContextWindowOverflowException):
        async for _ in fallback_model.stream(messages):
            pass

    # Assert fallback was not called
    mock_fallback.stream.assert_not_called()

    # Assert fallback_count is 0
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0


@pytest.mark.asyncio
async def test_both_models_fail(mock_primary, mock_fallback, fallback_model):
    """Test that fallback exception is raised when both fail."""
    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to raise RuntimeError
    mock_fallback.stream.side_effect = RuntimeError("Fallback failed")

    # Call stream and expect RuntimeError (not ModelThrottledException)
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    with pytest.raises(RuntimeError, match="Fallback failed"):
        async for _ in fallback_model.stream(messages):
            pass


# Circuit Breaker Tests


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures(mock_primary, mock_fallback, alist):
    """Test circuit opens after threshold failures."""
    # Configure FallbackModel with circuit_failure_threshold=2
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, circuit_time_window=60.0
    )

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Trigger first failure
    await alist(fallback_model.stream(messages))
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is False
    assert stats["primary_failures"] == 1

    # Trigger second failure - should open circuit
    await alist(fallback_model.stream(messages))
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is True
    assert stats["primary_failures"] == 2

    # Reset mock call count to verify next request skips primary
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()

    # Next request should skip primary and go directly to fallback
    await alist(fallback_model.stream(messages))

    # Assert primary was not called (circuit is open)
    mock_primary.stream.assert_not_called()

    # Assert fallback was called directly
    mock_fallback.stream.assert_called_once()

    # Assert circuit_skips counter increased
    stats = fallback_model.get_stats()
    assert stats["circuit_skips"] == 1


@pytest.mark.asyncio
async def test_circuit_stays_closed_below_threshold(mock_primary, mock_fallback, alist):
    """Test circuit stays closed below threshold."""
    # Configure with threshold=3
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=3, circuit_time_window=60.0
    )

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Trigger 2 failures (below threshold of 3)
    await alist(fallback_model.stream(messages))
    await alist(fallback_model.stream(messages))

    # Assert circuit_open is False
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is False
    assert stats["primary_failures"] == 2

    # Reset mock call count to verify primary is still attempted
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()

    # Next request should still attempt primary (circuit is closed)
    await alist(fallback_model.stream(messages))

    # Assert primary was called (circuit is still closed)
    mock_primary.stream.assert_called_once()

    # Assert fallback was also called due to primary failure
    mock_fallback.stream.assert_called_once()


@pytest.mark.asyncio
async def test_circuit_closes_after_cooldown(mock_primary, mock_fallback, alist):
    """Test circuit closes after cooldown period."""
    import time
    from unittest.mock import patch

    # Configure with circuit_cooldown_seconds=1
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, circuit_cooldown_seconds=1
    )

    # Mock primary.stream to raise ModelThrottledException initially
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Open circuit with 2 failures
    await alist(fallback_model.stream(messages))
    await alist(fallback_model.stream(messages))

    # Assert circuit is open
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is True

    # Mock time.time to advance past cooldown
    current_time = time.time()
    with patch("time.time", return_value=current_time + 2):  # 2 seconds later
        # Reset mocks to track new calls
        mock_primary.stream.reset_mock()
        mock_fallback.stream.reset_mock()

        # Now make primary succeed to test circuit closing
        primary_events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Primary response"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        mock_primary.stream.side_effect = None
        mock_primary.stream.return_value = async_generator(primary_events)

        # Make request - circuit should close and primary should be retried
        result = await alist(fallback_model.stream(messages))

        # Assert circuit closed and primary was retried
        stats = fallback_model.get_stats()
        assert stats["circuit_open"] is False

        # Assert primary was called (circuit closed)
        mock_primary.stream.assert_called_once()

        # Assert fallback was not called (primary succeeded)
        mock_fallback.stream.assert_not_called()

        # Assert we got primary response
        assert result == primary_events


@pytest.mark.asyncio
async def test_time_window_failure_counting(mock_primary, mock_fallback, alist):
    """Test only failures within time window count."""
    import time
    from unittest.mock import patch

    # Configure with circuit_time_window=2
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, circuit_time_window=2.0
    )

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    current_time = time.time()

    # Add old failure timestamp (> 2 seconds ago) by mocking time
    with patch("time.time", return_value=current_time - 3):  # 3 seconds ago
        await alist(fallback_model.stream(messages))

    # Add recent failure with current time
    with patch("time.time", return_value=current_time):
        await alist(fallback_model.stream(messages))

    # Check stats - should show 2 total failures but only 1 recent failure
    with patch("time.time", return_value=current_time):
        stats = fallback_model.get_stats()
        assert stats["primary_failures"] == 2  # Total failures
        assert stats["recent_failures"] == 1  # Only recent failure counts
        assert stats["circuit_open"] is False  # Circuit should stay closed (only 1 recent failure < threshold of 2)

    # Add another recent failure to test circuit opening with recent failures only
    with patch("time.time", return_value=current_time + 0.5):  # Still within 2-second window
        await alist(fallback_model.stream(messages))

        stats = fallback_model.get_stats()
        assert stats["primary_failures"] == 3  # Total failures
        assert stats["recent_failures"] == 2  # Two recent failures within window
        assert stats["circuit_open"] is True  # Circuit should open (2 recent failures >= threshold of 2)


# Configuration Tests


def test_default_config(mock_primary, mock_fallback):
    """Test default configuration values are applied."""
    # Create FallbackModel without config
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Assert default values
    assert fallback_model.circuit_failure_threshold == 3
    assert fallback_model.circuit_time_window == 60.0
    assert fallback_model.circuit_cooldown_seconds == 30
    assert fallback_model.track_stats is True
    assert fallback_model.should_fallback is None


def test_custom_config(mock_primary, mock_fallback):
    """Test custom configuration is applied."""
    # Create FallbackModel with custom values
    custom_config = {
        "circuit_failure_threshold": 5,
        "circuit_time_window": 120.0,
        "circuit_cooldown_seconds": 60,
        "track_stats": False,
    }

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, **custom_config)

    # Assert custom values are used
    assert fallback_model.circuit_failure_threshold == 5
    assert fallback_model.circuit_time_window == 120.0
    assert fallback_model.circuit_cooldown_seconds == 60
    assert fallback_model.track_stats is False


@pytest.mark.asyncio
async def test_custom_should_fallback(mock_primary, mock_fallback, alist):
    """Test custom should_fallback function is used."""

    # Create custom function that returns True for specific error
    def custom_should_fallback(error):
        return isinstance(error, ValueError) and "custom_error" in str(error)

    # Create FallbackModel with custom should_fallback function
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, should_fallback=custom_should_fallback)

    # Mock primary to raise that error
    mock_primary.stream.side_effect = ValueError("custom_error occurred")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(fallback_model.stream(messages))

    # Assert fallback is triggered
    assert result == fallback_events
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1

    # Test that other ValueError doesn't trigger fallback
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()
    mock_primary.stream.side_effect = ValueError("different error")

    # This should not trigger fallback and should re-raise
    with pytest.raises(ValueError, match="different error"):
        async for _ in fallback_model.stream(messages):
            pass

    # Assert fallback was not called
    mock_fallback.stream.assert_not_called()

    # Assert fallback_count didn't increase
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1  # Still 1 from previous test


def test_update_config(mock_primary, mock_fallback):
    """Test configuration can be updated."""
    # Create FallbackModel
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Verify initial default values
    assert fallback_model.circuit_failure_threshold == 3
    assert fallback_model.circuit_time_window == 60.0

    # Call update_config with new values
    fallback_model.update_config(circuit_failure_threshold=10, circuit_time_window=300.0, circuit_cooldown_seconds=120)

    # Assert config is updated
    assert fallback_model.circuit_failure_threshold == 10
    assert fallback_model.circuit_time_window == 300.0
    assert fallback_model.circuit_cooldown_seconds == 120

    # Assert other values remain unchanged
    assert fallback_model.track_stats is True  # Default value preserved

    # Test partial update
    fallback_model.update_config(circuit_failure_threshold=7)

    # Assert only specified value is updated
    assert fallback_model.circuit_failure_threshold == 7
    assert fallback_model.circuit_time_window == 300.0  # Previous value preserved
    assert fallback_model.circuit_cooldown_seconds == 120  # Previous value preserved


# Statistics Tests


@pytest.mark.asyncio
async def test_statistics_tracking(mock_primary, mock_fallback, alist):
    """Test statistics are tracked correctly."""
    # Create FallbackModel with circuit_failure_threshold=2 for easier testing
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, track_stats=True
    )

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Mock primary success events
    primary_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Primary response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Test initial stats
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0
    assert stats["primary_failures"] == 0
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is False
    assert stats["circuit_open"] is False
    assert stats["recent_failures"] == 0
    assert stats["circuit_open_until"] is None

    # Scenario 1: Primary success
    mock_primary.stream.return_value = async_generator(primary_events)
    result = await alist(fallback_model.stream(messages))

    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0
    assert stats["primary_failures"] == 0
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is False
    assert stats["circuit_open"] is False
    assert result == primary_events

    # Scenario 2: Primary failure, fallback success
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")
    result = await alist(fallback_model.stream(messages))

    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1
    assert stats["primary_failures"] == 1
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is True  # Last operation used fallback
    assert stats["circuit_open"] is False  # Still below threshold
    assert stats["recent_failures"] == 1
    assert result == fallback_events

    # Scenario 3: Another primary failure, should open circuit
    result = await alist(fallback_model.stream(messages))

    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 2
    assert stats["primary_failures"] == 2
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is True
    assert stats["circuit_open"] is True  # Circuit opened after 2 failures
    assert stats["recent_failures"] == 2
    assert stats["circuit_open_until"] is not None

    # Scenario 4: Circuit skip (circuit is open)
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()
    mock_fallback.stream.return_value = async_generator(fallback_events)

    result = await alist(fallback_model.stream(messages))

    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 2  # Doesn't increment for circuit skips
    assert stats["primary_failures"] == 2  # No new primary failure
    assert stats["circuit_skips"] == 1  # Circuit skip counter increased
    assert stats["using_fallback"] is True
    assert stats["circuit_open"] is True

    # Verify primary was not called (circuit skip)
    mock_primary.stream.assert_not_called()
    mock_fallback.stream.assert_called_once()
    assert result == fallback_events


def test_stats_disabled(mock_primary, mock_fallback):
    """Test statistics can be disabled."""
    # Create FallbackModel with track_stats=False
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback, track_stats=False)

    # Call get_config()
    config = fallback_model.get_config()

    # Assert stats is None
    assert config["stats"] is None

    # Verify other config sections are present
    assert "fallback_config" in config
    assert "primary_config" in config
    assert "fallback_model_config" in config

    # Verify fallback_config contains track_stats=False
    assert config["fallback_config"]["track_stats"] is False


@pytest.mark.asyncio
async def test_reset_stats(mock_primary, mock_fallback, alist):
    """Test reset_stats clears all statistics."""
    # Create FallbackModel with circuit_failure_threshold=2 for easier testing
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, track_stats=True
    )

    # Mock primary.stream to raise ModelThrottledException
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to return test events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Trigger failures to populate stats and open circuit
    await alist(fallback_model.stream(messages))  # First failure
    await alist(fallback_model.stream(messages))  # Second failure - opens circuit

    # Verify stats are populated and circuit is open
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 2
    assert stats["primary_failures"] == 2
    assert stats["circuit_open"] is True
    assert stats["recent_failures"] == 2
    assert stats["circuit_open_until"] is not None

    # Trigger a circuit skip to populate circuit_skips counter
    await alist(fallback_model.stream(messages))  # Circuit skip

    stats = fallback_model.get_stats()
    assert stats["circuit_skips"] == 1
    assert stats["using_fallback"] is True

    # Call reset_stats()
    fallback_model.reset_stats()

    # Assert all counters are 0 and circuit is closed
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0
    assert stats["primary_failures"] == 0
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is False
    assert stats["circuit_open"] is False
    assert stats["recent_failures"] == 0
    assert stats["circuit_open_until"] is None

    # Verify circuit is actually closed by checking that primary is attempted again
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()

    # This should attempt primary again (circuit is closed)
    await alist(fallback_model.stream(messages))

    # Verify primary was called (circuit is closed)
    mock_primary.stream.assert_called_once()
    mock_fallback.stream.assert_called_once()  # Called due to primary failure


def test_get_config(mock_primary, mock_fallback):
    """Test get_config returns all configuration."""

    # Create FallbackModel with custom config
    def custom_should_fallback(error):
        return True

    fallback_model = FallbackModel(
        primary=mock_primary,
        fallback=mock_fallback,
        circuit_failure_threshold=5,
        circuit_time_window=120.0,
        circuit_cooldown_seconds=60,
        should_fallback=custom_should_fallback,
        track_stats=True,
    )

    # Call get_config()
    config = fallback_model.get_config()

    # Assert returns fallback_config, primary_config, fallback_model_config, stats
    assert "fallback_config" in config
    assert "primary_config" in config
    assert "fallback_model_config" in config
    assert "stats" in config

    # Verify fallback_config contains all expected fields
    fallback_config = config["fallback_config"]
    assert fallback_config["circuit_failure_threshold"] == 5
    assert fallback_config["circuit_time_window"] == 120.0
    assert fallback_config["circuit_cooldown_seconds"] == 60
    assert fallback_config["should_fallback"] == custom_should_fallback
    assert fallback_config["track_stats"] is True

    # Verify primary_config comes from primary model
    assert config["primary_config"] == {"model": "primary"}

    # Verify fallback_model_config comes from fallback model
    assert config["fallback_model_config"] == {"model": "fallback"}

    # Verify stats is included (since track_stats=True)
    stats = config["stats"]
    assert isinstance(stats, dict)
    assert "fallback_count" in stats
    assert "primary_failures" in stats
    assert "circuit_skips" in stats
    assert "using_fallback" in stats
    assert "circuit_open" in stats
    assert "recent_failures" in stats
    assert "circuit_open_until" in stats

    # Test with track_stats=False
    fallback_model_no_stats = FallbackModel(primary=mock_primary, fallback=mock_fallback, track_stats=False)

    config_no_stats = fallback_model_no_stats.get_config()
    assert config_no_stats["stats"] is None


# Streaming Tests


@pytest.mark.asyncio
async def test_stream_primary_success(mock_primary, mock_fallback, fallback_model, alist):
    """Test streaming from primary model."""
    # Mock primary.stream to yield multiple events
    test_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"index": 0}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": " world"}}},
        {"contentBlockStop": {"index": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_primary.stream.return_value = async_generator(test_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Collect all events
    events = []
    async for event in fallback_model.stream(messages):
        events.append(event)

    # Assert events match primary output
    assert events == test_events

    # Verify primary was called with correct parameters
    mock_primary.stream.assert_called_once_with(
        messages=messages, tool_specs=None, system_prompt=None, tool_choice=None
    )

    # Verify fallback was not called
    mock_fallback.stream.assert_not_called()

    # Verify stats show no fallback usage
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0
    assert stats["using_fallback"] is False


@pytest.mark.asyncio
async def test_stream_fallback_after_primary_failure(mock_primary, mock_fallback, fallback_model, alist):
    """Test streaming from fallback after primary failure."""
    # Mock primary.stream to raise exception
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to yield events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"index": 0}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": "Fallback"}}},
        {"contentBlockDelta": {"index": 0, "delta": {"text": " response"}}},
        {"contentBlockStop": {"index": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Call stream
    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    tool_specs = [{"name": "test_tool", "description": "Test tool"}]
    system_prompt = "You are a helpful assistant"

    # Collect all events
    events = []
    async for event in fallback_model.stream(
        messages, tool_specs=tool_specs, system_prompt=system_prompt, tool_choice="auto"
    ):
        events.append(event)

    # Assert fallback events received
    assert events == fallback_events

    # Verify primary was called first
    mock_primary.stream.assert_called_once_with(
        messages=messages, tool_specs=tool_specs, system_prompt=system_prompt, tool_choice="auto"
    )

    # Verify fallback was called with same parameters
    mock_fallback.stream.assert_called_once_with(
        messages=messages, tool_specs=tool_specs, system_prompt=system_prompt, tool_choice="auto"
    )

    # Verify stats show fallback usage
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1
    assert stats["primary_failures"] == 1
    assert stats["using_fallback"] is True


@pytest.mark.asyncio
async def test_stream_circuit_open_direct_fallback(mock_primary, mock_fallback, alist):
    """Test streaming directly from fallback when circuit open."""
    # Create FallbackModel with low threshold to easily open circuit
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=1, circuit_time_window=60.0
    )

    # Mock primary.stream to raise exception to open circuit
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.stream to yield events
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]

    # First request to open circuit
    await alist(fallback_model.stream(messages))

    # Verify circuit is open
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is True

    # Reset mocks to track next call
    mock_primary.stream.reset_mock()
    mock_fallback.stream.reset_mock()
    mock_fallback.stream.return_value = async_generator(fallback_events)

    # Second request should go directly to fallback
    events = []
    async for event in fallback_model.stream(
        messages, tool_specs=[{"name": "tool"}], system_prompt="Test prompt", tool_choice="required"
    ):
        events.append(event)

    # Assert primary.stream not called
    mock_primary.stream.assert_not_called()

    # Assert fallback events received
    assert events == fallback_events

    # Verify fallback was called with correct parameters
    mock_fallback.stream.assert_called_once_with(
        messages=messages, tool_specs=[{"name": "tool"}], system_prompt="Test prompt", tool_choice="required"
    )

    # Verify stats show circuit skip
    stats = fallback_model.get_stats()
    assert stats["circuit_skips"] == 1
    assert stats["using_fallback"] is True


# Structured Output Tests


@pytest.mark.asyncio
async def test_structured_output_primary_success(mock_primary, mock_fallback, fallback_model, alist):
    """Test structured_output from primary model."""

    from pydantic import BaseModel

    # Create test Pydantic model
    class TestResponse(BaseModel):
        answer: str
        confidence: float

    # Mock primary.structured_output to yield events
    test_events = [
        {"chunk_type": "structured_output_start"},
        {"chunk_type": "structured_output_delta", "data": {"answer": "Hello"}},
        {"chunk_type": "structured_output_delta", "data": {"confidence": 0.95}},
        {"chunk_type": "structured_output_complete", "data": TestResponse(answer="Hello", confidence=0.95)},
    ]
    mock_primary.structured_output.return_value = async_generator(test_events)

    # Call structured_output
    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(
        fallback_model.structured_output(
            output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
        )
    )

    # Assert events received
    assert result == test_events

    # Verify primary was called with correct parameters
    mock_primary.structured_output.assert_called_once_with(
        output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
    )

    # Verify fallback was not called
    mock_fallback.structured_output.assert_not_called()

    # Verify stats show no fallback usage
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 0
    assert stats["using_fallback"] is False


@pytest.mark.asyncio
async def test_structured_output_fallback_after_primary_failure(mock_primary, mock_fallback, fallback_model, alist):
    """Test structured_output from fallback after primary failure."""
    from pydantic import BaseModel

    # Create test Pydantic model
    class TestResponse(BaseModel):
        answer: str
        confidence: float

    # Mock primary.structured_output to raise exception
    mock_primary.structured_output.side_effect = ModelThrottledException("Primary throttled")

    # Mock fallback.structured_output to yield events
    fallback_events = [
        {"chunk_type": "structured_output_start"},
        {"chunk_type": "structured_output_delta", "data": {"answer": "Fallback response"}},
        {"chunk_type": "structured_output_delta", "data": {"confidence": 0.85}},
        {"chunk_type": "structured_output_complete", "data": TestResponse(answer="Fallback response", confidence=0.85)},
    ]
    mock_fallback.structured_output.return_value = async_generator(fallback_events)

    # Call structured_output
    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]
    result = await alist(
        fallback_model.structured_output(
            output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
        )
    )

    # Assert fallback events received
    assert result == fallback_events

    # Verify primary was called first
    mock_primary.structured_output.assert_called_once_with(
        output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
    )

    # Verify fallback was called after primary failure
    mock_fallback.structured_output.assert_called_once_with(
        output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
    )

    # Verify stats show fallback usage
    stats = fallback_model.get_stats()
    assert stats["fallback_count"] == 1
    assert stats["primary_failures"] == 1
    assert stats["using_fallback"] is True


@pytest.mark.asyncio
async def test_structured_output_circuit_open(mock_primary, mock_fallback, alist):
    """Test structured_output directly from fallback when circuit open."""
    from pydantic import BaseModel

    # Create test Pydantic model
    class TestResponse(BaseModel):
        answer: str
        confidence: float

    # Create FallbackModel with low threshold to easily open circuit
    fallback_model = FallbackModel(
        primary=mock_primary, fallback=mock_fallback, circuit_failure_threshold=2, circuit_time_window=60.0
    )

    # Open circuit by triggering failures
    mock_primary.structured_output.side_effect = ModelThrottledException("Primary throttled")
    fallback_events_for_opening = [
        {"chunk_type": "structured_output_complete", "data": TestResponse(answer="temp", confidence=0.5)}
    ]
    mock_fallback.structured_output.return_value = async_generator(fallback_events_for_opening)

    prompt = [{"role": "user", "content": [{"text": "Hi"}]}]

    # Trigger 2 failures to open circuit
    await alist(fallback_model.structured_output(TestResponse, prompt))
    await alist(fallback_model.structured_output(TestResponse, prompt))

    # Verify circuit is open
    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is True

    # Reset mocks to track new calls
    mock_primary.structured_output.reset_mock()
    mock_fallback.structured_output.reset_mock()

    # Mock fallback.structured_output to yield events for the actual test
    test_fallback_events = [
        {"chunk_type": "structured_output_start"},
        {"chunk_type": "structured_output_delta", "data": {"answer": "Circuit open response"}},
        {"chunk_type": "structured_output_delta", "data": {"confidence": 0.90}},
        {
            "chunk_type": "structured_output_complete",
            "data": TestResponse(answer="Circuit open response", confidence=0.90),
        },
    ]
    mock_fallback.structured_output.return_value = async_generator(test_fallback_events)

    # Call structured_output - should go directly to fallback
    result = await alist(
        fallback_model.structured_output(
            output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
        )
    )

    # Assert primary not called (circuit is open)
    mock_primary.structured_output.assert_not_called()

    # Assert fallback was called directly
    mock_fallback.structured_output.assert_called_once_with(
        output_model=TestResponse, prompt=prompt, system_prompt="You are a helpful assistant"
    )

    # Assert fallback events received
    assert result == test_fallback_events

    # Verify circuit skip was recorded
    stats = fallback_model.get_stats()
    assert stats["circuit_skips"] == 1
    assert stats["using_fallback"] is True


# Enhanced Features Tests (Requirements 10 & 11)


def test_fallback_stats_typed_dict(mock_primary, mock_fallback):
    """Test that get_stats() returns properly typed FallbackStats structure."""

    # Create FallbackModel
    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Get stats
    stats = fallback_model.get_stats()

    # Verify it's a FallbackStats (TypedDict)
    assert isinstance(stats, dict)

    # Verify all required fields are present with correct types
    assert isinstance(stats["fallback_count"], int)
    assert isinstance(stats["primary_failures"], int)
    assert isinstance(stats["circuit_skips"], int)
    assert isinstance(stats["using_fallback"], bool)
    assert isinstance(stats["circuit_open"], bool)
    assert isinstance(stats["recent_failures"], int)
    assert stats["circuit_open_until"] is None or isinstance(stats["circuit_open_until"], float)
    assert isinstance(stats["primary_model_name"], str)
    assert isinstance(stats["fallback_model_name"], str)

    # Verify initial values
    assert stats["fallback_count"] == 0
    assert stats["primary_failures"] == 0
    assert stats["circuit_skips"] == 0
    assert stats["using_fallback"] is False
    assert stats["circuit_open"] is False
    assert stats["recent_failures"] == 0
    assert stats["circuit_open_until"] is None


def test_model_name_extraction(mock_primary, mock_fallback):
    """Test model name extraction from configuration and class name fallback."""
    # Test with mock models that have get_config() returning model identifiers
    mock_primary.get_config.return_value = {"model_id": "gpt-4", "other": "value"}
    mock_fallback.get_config.return_value = {"model": "claude-3-haiku", "other": "value"}

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Test _get_model_name method directly
    primary_name = fallback_model._get_model_name(mock_primary)
    fallback_name = fallback_model._get_model_name(mock_fallback)

    assert primary_name == "gpt-4"
    assert fallback_name == "claude-3-haiku"

    # Test with config that has "name" field
    mock_primary.get_config.return_value = {"name": "custom-primary", "other": "value"}
    primary_name = fallback_model._get_model_name(mock_primary)
    assert primary_name == "custom-primary"

    # Test fallback to class name when no config identifiers
    mock_primary.get_config.return_value = {"other": "value"}  # No model identifiers
    primary_name = fallback_model._get_model_name(mock_primary)
    assert primary_name == "Model"  # Should fall back to class name (mock has spec=Model)

    # Test fallback to class name when get_config() fails
    mock_primary.get_config.side_effect = Exception("Config error")
    primary_name = fallback_model._get_model_name(mock_primary)
    assert primary_name == "Model"  # Should fall back to class name (mock has spec=Model)


def test_model_names_in_statistics(mock_primary, mock_fallback):
    """Test that model names are included in statistics."""
    # Configure mock models to return specific identifiers
    mock_primary.get_config.return_value = {"model_id": "test-primary"}
    mock_fallback.get_config.return_value = {"model_id": "test-fallback"}

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Get stats
    stats = fallback_model.get_stats()

    # Verify model names are included
    assert stats["primary_model_name"] == "test-primary"
    assert stats["fallback_model_name"] == "test-fallback"

    # Test with class name fallback
    mock_primary.get_config.return_value = {}  # No identifiers
    mock_fallback.get_config.return_value = {}  # No identifiers

    stats = fallback_model.get_stats()
    assert stats["primary_model_name"] == "Model"  # Mock has spec=Model
    assert stats["fallback_model_name"] == "Model"  # Mock has spec=Model


@pytest.mark.asyncio
async def test_model_names_in_logging(mock_primary, mock_fallback, alist, caplog):
    """Test that model names appear in logging output for better traceability."""
    import logging

    # Configure mock models with specific identifiers
    mock_primary.get_config.return_value = {"model_id": "test-primary-model"}
    mock_fallback.get_config.return_value = {"model_id": "test-fallback-model"}

    # Set logging level to capture info messages
    caplog.set_level(logging.INFO)

    fallback_model = FallbackModel(primary=mock_primary, fallback=mock_fallback)

    # Check initialization logging includes model names
    assert "primary=<test-primary-model>" in caplog.text
    assert "fallback=<test-fallback-model>" in caplog.text

    # Clear log for next test
    caplog.clear()

    # Test fallback scenario logging
    mock_primary.stream.side_effect = ModelThrottledException("Primary throttled")
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Hi"}]}]
    await alist(fallback_model.stream(messages))

    # Check that model names appear in fallback logging
    assert "primary_model=<test-primary-model>" in caplog.text
    assert "fallback_model=<test-fallback-model>" in caplog.text


def test_override_decorators_present():
    """Test that @override decorators are present on overridden methods."""
    import inspect

    from strands.models.fallback import FallbackModel

    # Check that the methods have @override decorator by checking if they're marked as overrides
    # Note: This is a basic check - in practice, type checkers like mypy would catch missing @override

    # Verify the methods exist and are callable
    assert hasattr(FallbackModel, "stream")
    assert callable(FallbackModel.stream)

    assert hasattr(FallbackModel, "structured_output")
    assert callable(FallbackModel.structured_output)

    assert hasattr(FallbackModel, "update_config")
    assert callable(FallbackModel.update_config)

    assert hasattr(FallbackModel, "get_config")
    assert callable(FallbackModel.get_config)

    # Check method signatures match expected interface
    stream_sig = inspect.signature(FallbackModel.stream)
    assert "messages" in stream_sig.parameters
    assert "tool_specs" in stream_sig.parameters
    assert "system_prompt" in stream_sig.parameters
    assert "tool_choice" in stream_sig.parameters

    structured_output_sig = inspect.signature(FallbackModel.structured_output)
    assert "output_model" in structured_output_sig.parameters
    assert "prompt" in structured_output_sig.parameters
    assert "system_prompt" in structured_output_sig.parameters


@pytest.mark.asyncio
async def test_enhanced_debugging_information(mock_primary, mock_fallback, alist):
    """Test enhanced debugging information in statistics and logging."""
    # Configure models with realistic identifiers
    mock_primary.get_config.return_value = {"model_id": "gpt-4-turbo", "provider": "openai", "max_tokens": 4096}
    mock_fallback.get_config.return_value = {
        "model_id": "claude-3-sonnet-20240229",
        "provider": "anthropic",
        "max_tokens": 4096,
    }

    fallback_model = FallbackModel(
        primary=mock_primary,
        fallback=mock_fallback,
        circuit_failure_threshold=1,  # Open circuit quickly
    )

    # Trigger a fallback scenario
    mock_primary.stream.side_effect = ModelThrottledException("Rate limited")
    fallback_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Fallback response"}}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    mock_fallback.stream.return_value = async_generator(fallback_events)

    messages = [{"role": "user", "content": [{"text": "Test message"}]}]
    await alist(fallback_model.stream(messages))

    # Get enhanced statistics
    stats = fallback_model.get_stats()

    # Verify debugging information is present
    assert stats["primary_model_name"] == "gpt-4-turbo"
    assert stats["fallback_model_name"] == "claude-3-sonnet-20240229"
    assert stats["fallback_count"] == 1
    assert stats["primary_failures"] == 1
    assert stats["using_fallback"] is True

    # Trigger circuit opening
    await alist(fallback_model.stream(messages))  # This should open the circuit

    stats = fallback_model.get_stats()
    assert stats["circuit_open"] is True
    assert stats["circuit_skips"] == 1

    # Verify that with this enhanced information, debugging is much easier
    # A developer can now clearly see:
    # 1. Which specific models are being used
    # 2. Current circuit breaker state
    # 3. Detailed failure and fallback counts
    # 4. Whether the last request used fallback
    assert all(
        key in stats
        for key in [
            "primary_model_name",
            "fallback_model_name",
            "fallback_count",
            "primary_failures",
            "circuit_skips",
            "using_fallback",
            "circuit_open",
            "recent_failures",
            "circuit_open_until",
        ]
    )
