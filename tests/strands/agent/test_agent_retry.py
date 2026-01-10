"""Integration tests for Agent retry_strategy parameter."""

from unittest.mock import Mock

import pytest

from strands import Agent
from strands.agent.retry import ModelRetryStrategy, NoopRetryStrategy
from strands.types.exceptions import ModelThrottledException
from tests.fixtures.mocked_model_provider import MockedModelProvider

# Agent Retry Strategy Initialization Tests


def test_agent_with_default_retry_strategy():
    """Test that Agent uses ModelRetryStrategy by default when retry_strategy=None."""
    agent = Agent()

    # Should have a retry_strategy
    assert hasattr(agent, "retry_strategy")
    assert agent.retry_strategy is not None

    # Should be ModelRetryStrategy with default parameters
    assert isinstance(agent.retry_strategy, ModelRetryStrategy)
    assert agent.retry_strategy._max_attempts == 6
    assert agent.retry_strategy._initial_delay == 4
    assert agent.retry_strategy._max_delay == 240


def test_agent_with_custom_model_retry_strategy():
    """Test Agent initialization with custom ModelRetryStrategy parameters."""
    custom_strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    agent = Agent(retry_strategy=custom_strategy)

    assert agent.retry_strategy is custom_strategy
    assert agent.retry_strategy._max_attempts == 3
    assert agent.retry_strategy._initial_delay == 2
    assert agent.retry_strategy._max_delay == 60


def test_agent_with_noop_retry_strategy():
    """Test Agent initialization with NoopRetryStrategy."""
    noop_strategy = NoopRetryStrategy()
    agent = Agent(retry_strategy=noop_strategy)

    assert agent.retry_strategy is noop_strategy
    assert isinstance(agent.retry_strategy, NoopRetryStrategy)


def test_retry_strategy_registered_as_hook():
    """Test that retry_strategy is registered with the hook system."""
    custom_strategy = ModelRetryStrategy(max_attempts=3)
    agent = Agent(retry_strategy=custom_strategy)

    # Verify retry strategy callback is registered
    from strands.hooks import AfterModelCallEvent

    callbacks = list(agent.hooks.get_callbacks_for(AfterModelCallEvent(agent=agent, exception=None)))

    # Should have at least one callback (from retry strategy)
    assert len(callbacks) > 0

    # Verify one of the callbacks is from the retry strategy
    assert any(
        callback.__self__ is custom_strategy if hasattr(callback, "__self__") else False for callback in callbacks
    )


# Agent Retry Behavior Tests


@pytest.mark.asyncio
async def test_agent_retries_with_default_strategy(mock_sleep):
    """Test that Agent retries on throttling with default ModelRetryStrategy."""
    # Create a model that fails twice with throttling, then succeeds
    model = Mock()
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException"),
        ModelThrottledException("ThrottlingException"),
        MockedModelProvider([{"role": "assistant", "content": [{"text": "Success after retries"}]}]).stream([]),
    ]

    agent = Agent(model=model)

    result = agent.stream_async("test prompt")
    events = [event async for event in result]

    # Should have succeeded after retries - just check we got events
    assert len(events) > 0

    # Should have slept twice (for two retries)
    assert len(mock_sleep.sleep_calls) == 2
    # First retry: 4 seconds
    assert mock_sleep.sleep_calls[0] == 4
    # Second retry: 8 seconds (exponential backoff)
    assert mock_sleep.sleep_calls[1] == 8


@pytest.mark.asyncio
async def test_agent_no_retry_with_noop_strategy():
    """Test that Agent does not retry with NoopRetryStrategy."""
    # Create a model that always fails with throttling
    model = Mock()
    model.stream.side_effect = ModelThrottledException("ThrottlingException")

    agent = Agent(model=model, retry_strategy=NoopRetryStrategy())

    # Should raise exception immediately without retry
    with pytest.raises(ModelThrottledException):
        result = agent.stream_async("test prompt")
        # Consume the stream to trigger the exception
        _ = [event async for event in result]


@pytest.mark.asyncio
async def test_agent_respects_max_attempts(mock_sleep):
    """Test that Agent respects max_attempts in retry strategy."""
    # Create a model that always fails
    model = Mock()
    model.stream.side_effect = ModelThrottledException("ThrottlingException")

    # Use custom strategy with max 2 attempts
    custom_strategy = ModelRetryStrategy(max_attempts=2, initial_delay=1, max_delay=60)
    agent = Agent(model=model, retry_strategy=custom_strategy)

    with pytest.raises(ModelThrottledException):
        result = agent.stream_async("test prompt")
        _ = [event async for event in result]

    # Should have attempted max_attempts times, which means (max_attempts - 1) sleeps
    # Attempt 0: fail, sleep
    # Attempt 1: fail, no more attempts
    assert len(mock_sleep.sleep_calls) == 1


# Backwards Compatibility Tests


@pytest.mark.asyncio
async def test_event_loop_throttle_event_emitted(mock_sleep):
    """Test that EventLoopThrottleEvent is still emitted for backwards compatibility."""
    # Create a model that fails once with throttling, then succeeds
    model = Mock()
    model.stream.side_effect = [
        ModelThrottledException("ThrottlingException"),
        MockedModelProvider([{"role": "assistant", "content": [{"text": "Success"}]}]).stream([]),
    ]

    agent = Agent(model=model)

    result = agent.stream_async("test prompt")
    events = [event async for event in result]

    # Should have EventLoopThrottleEvent in the stream
    throttle_events = [e for e in events if "event_loop_throttled_delay" in e]
    assert len(throttle_events) > 0

    # Should have the correct delay value
    assert throttle_events[0]["event_loop_throttled_delay"] > 0


@pytest.mark.asyncio
async def test_no_throttle_event_with_noop_strategy():
    """Test that EventLoopThrottleEvent is not emitted with NoopRetryStrategy."""
    # Create a model that succeeds immediately
    model = MockedModelProvider([{"role": "assistant", "content": [{"text": "Success"}]}])

    agent = Agent(model=model, retry_strategy=NoopRetryStrategy())

    result = agent.stream_async("test prompt")
    events = [event async for event in result]

    # Should not have any EventLoopThrottleEvent
    throttle_events = [e for e in events if "event_loop_throttled_delay" in e]
    assert len(throttle_events) == 0
