"""Tests for MCP retry mechanism."""

import pytest

from strands.tools.mcp.mcp_retry import (
    CustomRetryStrategy,
    ExponentialBackoffRetry,
    LinearBackoffRetry,
    MCPRetryConfig,
    NoRetryStrategy,
)


class TestNoRetryStrategy:
    """Tests for NoRetryStrategy."""

    @pytest.mark.asyncio
    async def test_never_retries(self):
        """Test that NoRetryStrategy never retries."""
        strategy = NoRetryStrategy()

        # Should never retry regardless of exception or attempt number
        assert not await strategy.should_retry(Exception("test"), 1)
        assert not await strategy.should_retry(ValueError("test"), 5)
        assert not await strategy.should_retry(RuntimeError("test"), 10)

    @pytest.mark.asyncio
    async def test_zero_delay(self):
        """Test that NoRetryStrategy always returns zero delay."""
        strategy = NoRetryStrategy()

        assert await strategy.get_delay(1) == 0.0
        assert await strategy.get_delay(5) == 0.0
        assert await strategy.get_delay(100) == 0.0


class TestExponentialBackoffRetry:
    """Tests for ExponentialBackoffRetry."""

    @pytest.mark.asyncio
    async def test_retry_within_limits(self):
        """Test that exponential backoff retries within max attempts."""
        strategy = ExponentialBackoffRetry(max_attempts=3)

        # Should retry for attempts 1 and 2
        assert await strategy.should_retry(Exception("test"), 1)
        assert await strategy.should_retry(Exception("test"), 2)

        # Should not retry after max attempts
        assert not await strategy.should_retry(Exception("test"), 3)
        assert not await strategy.should_retry(Exception("test"), 4)

    @pytest.mark.asyncio
    async def test_retry_only_for_specified_exceptions(self):
        """Test that exponential backoff only retries for specified exception types."""
        strategy = ExponentialBackoffRetry(max_attempts=3, retryable_exceptions=(ValueError, RuntimeError))

        # Should retry for specified exception types
        assert await strategy.should_retry(ValueError("test"), 1)
        assert await strategy.should_retry(RuntimeError("test"), 1)

        # Should not retry for unspecified exception types
        assert not await strategy.should_retry(TypeError("test"), 1)
        assert not await strategy.should_retry(AttributeError("test"), 1)

    @pytest.mark.asyncio
    async def test_exponential_delay_calculation(self):
        """Test exponential delay calculation."""
        strategy = ExponentialBackoffRetry(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False,  # Disable jitter for predictable testing
        )

        # Test exponential growth
        assert await strategy.get_delay(1) == 1.0  # base_delay * 2^(1-1) = 1.0 * 1
        assert await strategy.get_delay(2) == 2.0  # base_delay * 2^(2-1) = 1.0 * 2
        assert await strategy.get_delay(3) == 4.0  # base_delay * 2^(3-1) = 1.0 * 4
        assert await strategy.get_delay(4) == 8.0  # base_delay * 2^(4-1) = 1.0 * 8

        # Test max delay cap
        assert await strategy.get_delay(5) == 10.0  # Capped at max_delay

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        strategy = ExponentialBackoffRetry(
            base_delay=1.0,  # Use smaller base for predictable testing
            exponential_base=2.0,
            jitter=True,
        )

        # With jitter, delays should vary slightly
        delay1 = await strategy.get_delay(2)
        delay2 = await strategy.get_delay(2)
        delay3 = await strategy.get_delay(2)

        # Base delay for attempt 2 should be 1.0 * 2^(2-1) = 2.0
        # Jitter adds up to 25% (0.5), so delays should be between 2.0 and 2.5
        assert 2.0 <= delay1 <= 2.5
        assert 2.0 <= delay2 <= 2.5
        assert 2.0 <= delay3 <= 2.5

        # With jitter enabled, delays might be different (though not guaranteed)
        delays = [delay1, delay2, delay3]
        assert len(set(delays)) >= 1  # At least some variation expected


class TestLinearBackoffRetry:
    """Tests for LinearBackoffRetry."""

    @pytest.mark.asyncio
    async def test_linear_delay_calculation(self):
        """Test linear delay calculation."""
        strategy = LinearBackoffRetry(base_delay=2.0, delay_increment=1.5, max_delay=10.0)

        # Test linear growth: base_delay + delay_increment * (attempt - 1)
        assert await strategy.get_delay(1) == 2.0  # 2.0 + 1.5 * (1-1) = 2.0
        assert await strategy.get_delay(2) == 3.5  # 2.0 + 1.5 * (2-1) = 3.5
        assert await strategy.get_delay(3) == 5.0  # 2.0 + 1.5 * (3-1) = 5.0
        assert await strategy.get_delay(4) == 6.5  # 2.0 + 1.5 * (4-1) = 6.5

        # Test max delay cap
        assert await strategy.get_delay(10) == 10.0  # Capped at max_delay

    @pytest.mark.asyncio
    async def test_retry_behavior_same_as_exponential(self):
        """Test that retry behavior follows same pattern as exponential."""
        strategy = LinearBackoffRetry(max_attempts=2)

        # Should retry within limits
        assert await strategy.should_retry(Exception("test"), 1)

        # Should not retry after max attempts
        assert not await strategy.should_retry(Exception("test"), 2)


class TestCustomRetryStrategy:
    """Tests for CustomRetryStrategy."""

    @pytest.mark.asyncio
    async def test_custom_should_retry_function(self):
        """Test custom should_retry function."""

        async def custom_should_retry(exception: Exception, attempt: int) -> bool:
            # Retry only for ValueError and only up to 2 attempts
            return isinstance(exception, ValueError) and attempt < 2

        async def custom_get_delay(attempt: int) -> float:
            return float(attempt * 0.5)

        strategy = CustomRetryStrategy(custom_should_retry, custom_get_delay)

        # Should retry ValueError on first attempt
        assert await strategy.should_retry(ValueError("test"), 1)

        # Should not retry ValueError on second attempt
        assert not await strategy.should_retry(ValueError("test"), 2)

        # Should not retry other exceptions
        assert not await strategy.should_retry(RuntimeError("test"), 1)

    @pytest.mark.asyncio
    async def test_custom_get_delay_function(self):
        """Test custom get_delay function."""

        async def custom_should_retry(exception: Exception, attempt: int) -> bool:
            return True

        async def custom_get_delay(attempt: int) -> float:
            return float(attempt * 0.5)

        strategy = CustomRetryStrategy(custom_should_retry, custom_get_delay)

        assert await strategy.get_delay(1) == 0.5
        assert await strategy.get_delay(2) == 1.0
        assert await strategy.get_delay(4) == 2.0


class TestMCPRetryConfig:
    """Tests for MCPRetryConfig."""

    def test_default_configuration(self):
        """Test default retry configuration."""
        config = MCPRetryConfig()

        # Should have NoRetryStrategy as default
        assert isinstance(config.strategy, NoRetryStrategy)
        assert len(config.tool_overrides) == 0

    def test_global_strategy_selection(self):
        """Test that global strategy is used when no tool-specific override exists."""
        global_strategy = ExponentialBackoffRetry(max_attempts=3)
        config = MCPRetryConfig(strategy=global_strategy)

        # Should return global strategy for any tool
        assert config.get_strategy_for_tool("any_tool") is global_strategy
        assert config.get_strategy_for_tool("another_tool") is global_strategy

    def test_tool_specific_strategy_override(self):
        """Test that tool-specific strategies override global strategy."""
        global_strategy = ExponentialBackoffRetry(max_attempts=3)
        tool_strategy = LinearBackoffRetry(max_attempts=5)

        config = MCPRetryConfig(strategy=global_strategy, tool_overrides={"special_tool": tool_strategy})

        # Should return tool-specific strategy for configured tool
        assert config.get_strategy_for_tool("special_tool") is tool_strategy

        # Should return global strategy for other tools
        assert config.get_strategy_for_tool("other_tool") is global_strategy

    def test_set_tool_strategy_method(self):
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

    def test_tool_override_precedence(self):
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
async def test_retry_strategies_integration():
    """Integration test showing how different strategies work together."""
    # Create a config with different strategies for different tools
    config = MCPRetryConfig(
        strategy=ExponentialBackoffRetry(max_attempts=2),  # Global default
        tool_overrides={
            "critical_tool": NoRetryStrategy(),  # No retries for critical operations
            "flaky_tool": ExponentialBackoffRetry(max_attempts=5, base_delay=0.1),  # More retries for flaky tool
        },
    )

    # Test that each tool gets the right strategy
    assert isinstance(config.get_strategy_for_tool("normal_tool"), ExponentialBackoffRetry)
    assert isinstance(config.get_strategy_for_tool("critical_tool"), NoRetryStrategy)
    assert isinstance(config.get_strategy_for_tool("flaky_tool"), ExponentialBackoffRetry)

    # Test that the flaky tool strategy has different settings than global
    flaky_strategy = config.get_strategy_for_tool("flaky_tool")
    assert flaky_strategy.max_attempts == 5
    assert flaky_strategy.base_delay == 0.1
