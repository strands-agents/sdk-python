"""Unit tests for retry strategy implementations."""

from unittest.mock import Mock

import pytest

from strands.agent.retry import ModelRetryStrategy, NoopRetryStrategy
from strands.hooks import AfterModelCallEvent, HookRegistry
from strands.types.exceptions import ModelThrottledException

# ModelRetryStrategy Tests


def test_model_retry_strategy_init_with_defaults():
    """Test ModelRetryStrategy initialization with default parameters."""
    strategy = ModelRetryStrategy()
    assert strategy._max_attempts == 6
    assert strategy._initial_delay == 4
    assert strategy._max_delay == 240
    assert strategy._current_attempt == 0


def test_model_retry_strategy_init_with_custom_parameters():
    """Test ModelRetryStrategy initialization with custom parameters."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    assert strategy._max_attempts == 3
    assert strategy._initial_delay == 2
    assert strategy._max_delay == 60
    assert strategy._current_attempt == 0


def test_model_retry_strategy_calculate_delay_with_different_attempts():
    """Test _calculate_delay returns correct exponential backoff for different attempt numbers."""
    strategy = ModelRetryStrategy(initial_delay=2, max_delay=32)

    # Test exponential backoff: 2 * (2^attempt)
    assert strategy._calculate_delay(0) == 2  # 2 * 2^0 = 2
    assert strategy._calculate_delay(1) == 4  # 2 * 2^1 = 4
    assert strategy._calculate_delay(2) == 8  # 2 * 2^2 = 8
    assert strategy._calculate_delay(3) == 16  # 2 * 2^3 = 16
    assert strategy._calculate_delay(4) == 32  # 2 * 2^4 = 32 (at max)
    assert strategy._calculate_delay(5) == 32  # 2 * 2^5 = 64, capped at 32
    assert strategy._calculate_delay(10) == 32  # Large attempt, still capped


def test_model_retry_strategy_calculate_delay_respects_max_delay():
    """Test _calculate_delay respects max_delay cap."""
    strategy = ModelRetryStrategy(initial_delay=10, max_delay=50)

    assert strategy._calculate_delay(0) == 10  # 10 * 2^0 = 10
    assert strategy._calculate_delay(1) == 20  # 10 * 2^1 = 20
    assert strategy._calculate_delay(2) == 40  # 10 * 2^2 = 40
    assert strategy._calculate_delay(3) == 50  # 10 * 2^3 = 80, capped at 50
    assert strategy._calculate_delay(4) == 50  # 10 * 2^4 = 160, capped at 50


def test_model_retry_strategy_register_hooks():
    """Test that ModelRetryStrategy registers AfterModelCallEvent callback."""
    strategy = ModelRetryStrategy()
    registry = HookRegistry()

    strategy.register_hooks(registry)

    # Verify callback was registered
    assert AfterModelCallEvent in registry._registered_callbacks
    assert len(registry._registered_callbacks[AfterModelCallEvent]) == 1


@pytest.mark.asyncio
async def test_model_retry_strategy_retry_on_throttle_exception_first_attempt(mock_sleep):
    """Test retry behavior on first ModelThrottledException."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )

    await strategy._handle_after_model_call(event)

    # Should set retry to True
    assert event.retry is True
    # Should sleep for initial_delay (attempt 0: 2 * 2^0 = 2)
    assert mock_sleep.sleep_calls == [2]
    assert mock_sleep.sleep_calls[0] == strategy._calculate_delay(0)
    # Should increment attempt
    assert strategy._current_attempt == 1


@pytest.mark.asyncio
async def test_model_retry_strategy_exponential_backoff(mock_sleep):
    """Test exponential backoff calculation."""
    strategy = ModelRetryStrategy(max_attempts=5, initial_delay=2, max_delay=16)
    mock_agent = Mock()

    # Simulate multiple retries
    for _ in range(4):
        event = AfterModelCallEvent(
            agent=mock_agent,
            exception=ModelThrottledException("Throttled"),
        )
        await strategy._handle_after_model_call(event)
        assert event.retry is True

    # Verify exponential backoff with max_delay cap
    # attempt 0: 2*2^0=2, attempt 1: 2*2^1=4, attempt 2: 2*2^2=8, attempt 3: 2*2^3=16 (capped)
    assert mock_sleep.sleep_calls == [2, 4, 8, 16]
    for i, sleep_delay in enumerate(mock_sleep.sleep_calls):
        assert sleep_delay == strategy._calculate_delay(i)


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_after_max_attempts(mock_sleep):
    """Test that retry is not set after reaching max_attempts."""
    strategy = ModelRetryStrategy(max_attempts=2, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    # First attempt
    event1 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event1)
    assert event1.retry is True
    assert strategy._current_attempt == 1

    # Second attempt (at max_attempts)
    event2 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event2)
    # Should NOT retry after reaching max_attempts
    assert event2.retry is False
    assert strategy._current_attempt == 2


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_on_non_throttle_exception():
    """Test that retry is not set for non-throttling exceptions."""
    strategy = ModelRetryStrategy()
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ValueError("Some other error"),
    )

    await strategy._handle_after_model_call(event)

    # Should not retry on non-throttling exceptions
    assert event.retry is False
    assert strategy._current_attempt == 0


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_on_success():
    """Test that retry is not set when model call succeeds."""
    strategy = ModelRetryStrategy()
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Success"}]},
            stop_reason="end_turn",
        ),
    )

    await strategy._handle_after_model_call(event)

    # Should not retry on success
    assert event.retry is False


@pytest.mark.asyncio
async def test_model_retry_strategy_reset_on_success(mock_sleep):
    """Test that strategy resets attempt counter on successful call."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    # First failure
    event1 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event1)
    assert event1.retry is True
    assert strategy._current_attempt == 1
    # Should sleep for initial_delay (attempt 0: 2 * 2^0 = 2)
    assert mock_sleep.sleep_calls == [2]
    assert mock_sleep.sleep_calls[0] == strategy._calculate_delay(0)

    # Success - should reset
    event2 = AfterModelCallEvent(
        agent=mock_agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Success"}]},
            stop_reason="end_turn",
        ),
    )
    await strategy._handle_after_model_call(event2)
    assert event2.retry is False
    # Should reset to initial state
    assert strategy._current_attempt == 0
    assert strategy._calculate_delay(0) == 2


# NoopRetryStrategy Tests


def test_noop_retry_strategy_register_hooks_does_nothing():
    """Test that NoopRetryStrategy does not register any callbacks."""
    strategy = NoopRetryStrategy()
    registry = HookRegistry()

    strategy.register_hooks(registry)

    # Verify no callbacks were registered
    assert len(registry._registered_callbacks) == 0


@pytest.mark.asyncio
async def test_noop_retry_strategy_no_retry_on_throttle_exception():
    """Test that NoopRetryStrategy does not retry on throttle exceptions."""
    strategy = NoopRetryStrategy()
    registry = HookRegistry()
    strategy.register_hooks(registry)

    mock_agent = Mock()
    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )

    # Invoke callbacks (should be none registered)
    await registry.invoke_callbacks_async(event)

    # event.retry should still be False (default)
    assert event.retry is False
