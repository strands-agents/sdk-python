"""Tests for MCP retry mechanism using tenacity."""

import pytest
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from strands.tools.mcp.mcp_retry import (
    CustomRetryStrategy,
    ExponentialBackoffRetry,
    LinearBackoffRetry,
    MCPRetryConfig,
    NoRetryStrategy,
)


class TestNoRetryStrategy:
    """Tests for NoRetryStrategy."""

    def test_creates_retryer(self) -> None:
        """Test that NoRetryStrategy creates a valid retryer."""
        strategy = NoRetryStrategy()
        assert strategy.retryer is not None
        assert isinstance(strategy.retryer, AsyncRetrying)

    def test_strategy_name(self) -> None:
        """Test that strategy returns correct name."""
        strategy = NoRetryStrategy()
        assert strategy.get_strategy_name() == "NoRetryStrategy"

    @pytest.mark.asyncio
    async def test_no_retries_performed(self) -> None:
        """Test that NoRetryStrategy doesn't retry on failures."""
        strategy = NoRetryStrategy()
        attempt_count = 0

        async def failing_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Test error")

        # Should only attempt once
        with pytest.raises(ValueError):
            async for attempt in strategy.retryer:
                with attempt:
                    await failing_operation()

        assert attempt_count == 1


class TestExponentialBackoffRetry:
    """Tests for ExponentialBackoffRetry."""

    def test_creates_retryer_with_default_params(self) -> None:
        """Test that ExponentialBackoffRetry creates a valid retryer with defaults."""
        strategy = ExponentialBackoffRetry()
        assert strategy.retryer is not None
        assert strategy.max_attempts == 3

    def test_creates_retryer_with_custom_params(self) -> None:
        """Test that ExponentialBackoffRetry accepts custom parameters."""
        strategy = ExponentialBackoffRetry(
            max_attempts=5,
            multiplier=2.0,
            min_wait=0.5,
            max_wait=30.0,
            jitter=False,
        )
        assert strategy.max_attempts == 5
        assert strategy.multiplier == 2.0
        assert strategy.min_wait == 0.5
        assert strategy.max_wait == 30.0
        assert strategy.jitter is False

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Test that exponential backoff retries on failures."""
        strategy = ExponentialBackoffRetry(max_attempts=3)
        attempt_count = 0

        async def failing_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            async for attempt in strategy.retryer:
                with attempt:
                    await failing_operation()

        # Should attempt 3 times (initial + 2 retries)
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self) -> None:
        """Test that operation succeeds after retries."""
        strategy = ExponentialBackoffRetry(max_attempts=3)
        attempt_count = 0

        async def flaky_operation() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = None
        async for attempt in strategy.retryer:
            with attempt:
                result = await flaky_operation()

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_only_retries_specified_exceptions(self) -> None:
        """Test that only specified exception types are retried."""
        strategy = ExponentialBackoffRetry(
            max_attempts=3,
            retryable_exceptions=(ValueError,),
        )

        # ValueError should be retried
        attempt_count = 0

        async def value_error_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            async for attempt in strategy.retryer:
                with attempt:
                    await value_error_operation()

        assert attempt_count == 3  # All attempts used

        # TypeError should not be retried
        attempt_count = 0

        async def type_error_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise TypeError("Test error")

        with pytest.raises(TypeError):
            async for attempt in strategy.retryer:
                with attempt:
                    await type_error_operation()

        assert attempt_count == 1  # No retries


class TestLinearBackoffRetry:
    """Tests for LinearBackoffRetry."""

    def test_creates_retryer_with_default_params(self) -> None:
        """Test that LinearBackoffRetry creates a valid retryer with defaults."""
        strategy = LinearBackoffRetry()
        assert strategy.retryer is not None
        assert strategy.max_attempts == 3

    def test_creates_retryer_with_custom_params(self) -> None:
        """Test that LinearBackoffRetry accepts custom parameters."""
        strategy = LinearBackoffRetry(
            max_attempts=5,
            wait_time=2.0,
        )
        assert strategy.max_attempts == 5
        assert strategy.wait_time == 2.0

    @pytest.mark.asyncio
    async def test_retries_with_fixed_delay(self) -> None:
        """Test that linear backoff retries with fixed delay."""
        strategy = LinearBackoffRetry(max_attempts=3, wait_time=0.1)
        attempt_count = 0

        async def failing_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            async for attempt in strategy.retryer:
                with attempt:
                    await failing_operation()

        assert attempt_count == 3


class TestCustomRetryStrategy:
    """Tests for CustomRetryStrategy."""

    @pytest.mark.asyncio
    async def test_custom_retryer(self) -> None:
        """Test custom retry strategy with user-defined retryer."""
        # Create a custom retryer that retries ValueError up to 2 times
        custom_retryer = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_fixed(0.1),
            retry=retry_if_exception_type(ValueError),
            reraise=True,
        )

        strategy = CustomRetryStrategy(custom_retryer)
        attempt_count = 0

        async def failing_operation() -> None:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            async for attempt in strategy.retryer:
                with attempt:
                    await failing_operation()

        assert attempt_count == 2


class TestMCPRetryConfig:
    """Tests for MCPRetryConfig."""

    def test_default_configuration(self) -> None:
        """Test default retry configuration."""
        config = MCPRetryConfig()

        # Should have NoRetryStrategy as default
        assert isinstance(config.strategy, NoRetryStrategy)
        assert len(config.tool_overrides) == 0

    def test_global_strategy_selection(self) -> None:
        """Test that global strategy is used when no tool-specific override exists."""
        global_strategy = ExponentialBackoffRetry(max_attempts=3)
        config = MCPRetryConfig(strategy=global_strategy)

        # Should return global strategy for any tool
        assert config.get_strategy_for_tool("any_tool") is global_strategy
        assert config.get_strategy_for_tool("another_tool") is global_strategy

    def test_tool_specific_strategy_override(self) -> None:
        """Test that tool-specific strategies override global strategy."""
        global_strategy = ExponentialBackoffRetry(max_attempts=3)
        tool_strategy = LinearBackoffRetry(max_attempts=5)

        config = MCPRetryConfig(strategy=global_strategy, tool_overrides={"special_tool": tool_strategy})

        # Should return tool-specific strategy for configured tool
        assert config.get_strategy_for_tool("special_tool") is tool_strategy

        # Should return global strategy for other tools
        assert config.get_strategy_for_tool("other_tool") is global_strategy

    def test_set_tool_strategy_method(self) -> None:
        """Test set_tool_strategy method and method chaining."""
        config = MCPRetryConfig()
        strategy1 = ExponentialBackoffRetry(max_attempts=2)
        strategy2 = LinearBackoffRetry(max_attempts=4)

        # Test method chaining
        result = config.set_tool_strategy("tool1", strategy1).set_tool_strategy("tool2", strategy2)
        assert result is config  # Should return same instance for chaining

        # Test that strategies were set correctly
        assert config.get_strategy_for_tool("tool1") is strategy1
        assert config.get_strategy_for_tool("tool2") is strategy2

    def test_tool_override_precedence(self) -> None:
        """Test that tool overrides take precedence over global strategy."""
        global_strategy = NoRetryStrategy()
        override_strategy = ExponentialBackoffRetry(max_attempts=3)

        config = MCPRetryConfig(strategy=global_strategy)

        # Initially should use global strategy
        assert config.get_strategy_for_tool("test_tool") is global_strategy

        # After setting override, should use override strategy
        config.set_tool_strategy("test_tool", override_strategy)
        assert config.get_strategy_for_tool("test_tool") is override_strategy

        # Other tools should still use global strategy
        assert config.get_strategy_for_tool("other_tool") is global_strategy


@pytest.mark.asyncio
async def test_retry_strategies_integration() -> None:
    """Integration test showing how different strategies work together."""
    # Create a config with different strategies for different tools
    config = MCPRetryConfig(
        strategy=ExponentialBackoffRetry(max_attempts=2),  # Global default
        tool_overrides={
            "critical_tool": NoRetryStrategy(),  # No retries for critical operations
            "flaky_tool": ExponentialBackoffRetry(max_attempts=5, min_wait=0.1),  # More retries for flaky tool
        },
    )

    # Test that each tool gets the right strategy
    assert isinstance(config.get_strategy_for_tool("normal_tool"), ExponentialBackoffRetry)
    assert isinstance(config.get_strategy_for_tool("critical_tool"), NoRetryStrategy)
    assert isinstance(config.get_strategy_for_tool("flaky_tool"), ExponentialBackoffRetry)

    # Test that the flaky tool strategy has different settings than global
    flaky_strategy = config.get_strategy_for_tool("flaky_tool")
    assert flaky_strategy.max_attempts == 5
    assert flaky_strategy.min_wait == 0.1
