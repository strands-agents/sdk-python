"""MCP Client retry mechanism implementation.

This module provides retry strategies for MCP tool calls to handle transient failures
in a cost-effective manner. Users can configure retry behavior at both global and
per-tool levels to optimize for their specific use cases.
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MCPRetryStrategy(ABC):
    """Abstract base class for MCP retry strategies.

    Retry strategies determine whether a failed tool call should be retried
    and how long to wait between attempts. This allows for flexible retry
    behavior that can be customized based on specific error conditions,
    tool types, or business requirements.
    """

    @abstractmethod
    async def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the operation should be retried after a failure.

        Args:
            exception: The exception that caused the failure
            attempt: The current attempt number (1-based, so 1 = first attempt)

        Returns:
            bool: True if the operation should be retried, False otherwise
        """
        pass

    @abstractmethod
    async def get_delay(self, attempt: int) -> float:
        """Calculate the delay in seconds before the next retry attempt.

        Args:
            attempt: The current attempt number (1-based)

        Returns:
            float: Delay in seconds before the next attempt
        """
        pass


class NoRetryStrategy(MCPRetryStrategy):
    """A retry strategy that never retries.

    This is the default strategy that maintains backwards compatibility
    by disabling all retry behavior.
    """

    async def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Never retry any operation."""
        return False

    async def get_delay(self, attempt: int) -> float:
        """Return zero delay since no retries occur."""
        return 0.0


class ExponentialBackoffRetry(MCPRetryStrategy):
    """Exponential backoff retry strategy with jitter.

    This strategy implements exponential backoff with optional jitter to
    avoid the thundering herd problem. Delays increase exponentially
    with each retry attempt.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None,
    ):
        """Initialize the exponential backoff retry strategy.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds (cap for exponential growth)
            exponential_base: Base for exponential calculation (default 2.0)
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that are retryable.
                If None, all exceptions are considered retryable.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    async def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception is retryable and within attempt limits."""
        if attempt >= self.max_attempts:
            return False

        return isinstance(exception, self.retryable_exceptions)

    async def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(self.base_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)

        if self.jitter:
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


class LinearBackoffRetry(MCPRetryStrategy):
    """Linear backoff retry strategy.

    This strategy implements linear backoff where delays increase
    linearly with each retry attempt.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        delay_increment: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: Optional[tuple] = None,
    ):
        """Initialize the linear backoff retry strategy.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            delay_increment: Amount to increase delay by each attempt
            max_delay: Maximum delay in seconds
            retryable_exceptions: Tuple of exception types that are retryable.
                If None, all exceptions are considered retryable.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.delay_increment = delay_increment
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    async def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception is retryable and within attempt limits."""
        if attempt >= self.max_attempts:
            return False

        return isinstance(exception, self.retryable_exceptions)

    async def get_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        delay = self.base_delay + (self.delay_increment * (attempt - 1))
        return min(delay, self.max_delay)


class CustomRetryStrategy(MCPRetryStrategy):
    """Custom retry strategy that allows user-defined retry logic.

    This strategy provides maximum flexibility by allowing users to
    define their own retry conditions and delay calculations through
    callback functions.
    """

    def __init__(
        self,
        should_retry_func: Callable[[Exception, int], Awaitable[bool]],
        get_delay_func: Callable[[int], Awaitable[float]],
    ):
        """Initialize the custom retry strategy.

        Args:
            should_retry_func: Async function that takes (exception, attempt)
                and returns bool
            get_delay_func: Async function that takes (attempt) and returns float
        """
        self._should_retry_func = should_retry_func
        self._get_delay_func = get_delay_func

    async def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Delegate to user-defined should_retry function."""
        return await self._should_retry_func(exception, attempt)

    async def get_delay(self, attempt: int) -> float:
        """Delegate to user-defined get_delay function."""
        return await self._get_delay_func(attempt)


@dataclass
class MCPRetryConfig:
    """Configuration for MCP client retry behavior.

    This configuration allows setting a global retry strategy for the entire
    MCP client as well as tool-specific retry strategies that override the
    global strategy for particular tools.
    """

    strategy: MCPRetryStrategy = field(default_factory=NoRetryStrategy)
    """Global retry strategy applied to all tools unless overridden."""

    tool_overrides: Dict[str, MCPRetryStrategy] = field(default_factory=dict)
    """Tool-specific retry strategies that override the global strategy."""

    def get_strategy_for_tool(self, tool_name: str) -> MCPRetryStrategy:
        """Get the appropriate retry strategy for a specific tool.

        Args:
            tool_name: Name of the tool to get strategy for

        Returns:
            MCPRetryStrategy: The retry strategy to use (tool-specific or global)
        """
        return self.tool_overrides.get(tool_name, self.strategy)

    def set_tool_strategy(self, tool_name: str, strategy: MCPRetryStrategy) -> "MCPRetryConfig":
        """Set a retry strategy for a specific tool.

        Args:
            tool_name: Name of the tool to configure
            strategy: The retry strategy to use for this tool

        Returns:
            MCPRetryConfig: This configuration object for method chaining
        """
        self.tool_overrides[tool_name] = strategy
        return self
