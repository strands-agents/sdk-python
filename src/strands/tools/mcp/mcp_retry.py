"""MCP Client retry mechanism implementation using tenacity.

This module provides retry strategies for MCP tool calls to handle transient failures
in a cost-effective manner. Users can configure retry behavior at both global and
per-tool levels to optimize for their specific use cases.

This implementation wraps the tenacity library to provide a convenient interface
for MCP-specific retry patterns.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_none,
)
from tenacity.wait import wait_base

logger = logging.getLogger(__name__)


class MCPRetryStrategy:
    """Base class for MCP retry strategies.

    This wraps tenacity's AsyncRetrying to provide a convenient interface
    for configuring retry behavior in MCP tool calls.
    """

    def __init__(self, retryer: AsyncRetrying):
        """Initialize with a tenacity AsyncRetrying instance.

        Args:
            retryer: The tenacity AsyncRetrying instance to use
        """
        self._retryer = retryer

    @property
    def retryer(self) -> AsyncRetrying:
        """Get the underlying tenacity retryer."""
        return self._retryer

    def get_strategy_name(self) -> str:
        """Get a human-readable name for this strategy."""
        return self.__class__.__name__


class NoRetryStrategy(MCPRetryStrategy):
    """A retry strategy that never retries.

    This is the default strategy that maintains backwards compatibility
    by disabling all retry behavior.
    """

    def __init__(self) -> None:
        """Initialize the no-retry strategy."""
        super().__init__(
            AsyncRetrying(
                stop=stop_after_attempt(1),  # Only one attempt, no retries
                wait=wait_none(),
                reraise=True,
            )
        )


class ExponentialBackoffRetry(MCPRetryStrategy):
    """Exponential backoff retry strategy with jitter.

    This strategy implements exponential backoff with optional jitter to
    avoid the thundering herd problem. Delays increase exponentially
    with each retry attempt.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        multiplier: float = 1.0,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    ):
        """Initialize the exponential backoff retry strategy.

        Args:
            max_attempts: Maximum number of retry attempts (includes initial attempt)
            multiplier: Multiplier for the exponential backoff
            min_wait: Minimum wait time in seconds
            max_wait: Maximum wait time in seconds
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that are retryable.
                If None, all exceptions are considered retryable.
        """
        self.max_attempts = max_attempts
        self.multiplier = multiplier
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

        retry_condition = retry_if_exception_type(self.retryable_exceptions)

        wait_strategy: wait_base
        if jitter:
            wait_strategy = wait_exponential(
                multiplier=multiplier,
                min=min_wait,
                max=max_wait,
            )
        else:
            wait_strategy = wait_exponential(
                multiplier=multiplier,
                min=min_wait,
                max=max_wait,
                exp_base=2,
            )

        super().__init__(
            AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_strategy,
                retry=retry_condition,
                reraise=True,
            )
        )


class LinearBackoffRetry(MCPRetryStrategy):
    """Linear backoff retry strategy.

    This strategy implements linear backoff where delays increase
    linearly with each retry attempt.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        wait_time: float = 1.0,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    ):
        """Initialize the linear backoff retry strategy.

        Args:
            max_attempts: Maximum number of retry attempts (includes initial attempt)
            wait_time: Fixed wait time in seconds between retries
            retryable_exceptions: Tuple of exception types that are retryable.
                If None, all exceptions are considered retryable.
        """
        self.max_attempts = max_attempts
        self.wait_time = wait_time
        self.retryable_exceptions = retryable_exceptions or (Exception,)

        retry_condition = retry_if_exception_type(self.retryable_exceptions)

        super().__init__(
            AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_fixed(wait_time),
                retry=retry_condition,
                reraise=True,
            )
        )


class CustomRetryStrategy(MCPRetryStrategy):
    """Custom retry strategy that allows user-defined retry logic.

    This strategy provides maximum flexibility by allowing users to
    pass their own tenacity AsyncRetrying instance.
    """

    def __init__(self, retryer: AsyncRetrying):
        """Initialize the custom retry strategy.

        Args:
            retryer: A tenacity AsyncRetrying instance with custom configuration
        """
        super().__init__(retryer)


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
