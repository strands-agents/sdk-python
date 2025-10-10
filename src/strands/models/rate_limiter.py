"""Rate limiting utilities for model providers.

This module provides rate limiting capabilities for model invocations using a token bucket
algorithm to ensure compliance with provider API limits.

The implementation is currently synchronous but structured to facilitate future async support.
Key methods like _create_lock(), _sleep(), and _current_time() are isolated to minimize
changes needed when adding async compatibility. Once async support is added, these methods
will use asyncio equivalents (asyncio.Lock, asyncio.sleep, etc.) and model methods will
support both sync and async operation.
"""

import logging
import threading
import time
import weakref
from typing import Any, Generic, Optional, Protocol, Type, TypeVar, Union, overload, runtime_checkable

from typing_extensions import TypedDict, cast

logger = logging.getLogger(__name__)


@runtime_checkable
class RateLimitableModel(Protocol):
    """Protocol defining model methods that make API calls and need rate limiting."""

    def converse(self, messages: Any, tool_specs: Optional[Any] = None, system_prompt: Optional[str] = None) -> Any:
        """Send messages to the model and get response."""
        ...

    def stream(self, request: dict[str, Any]) -> Any:
        """Stream responses from the model."""
        ...

    def structured_output(self, output_model: Any, prompt: Any) -> Any:
        """Get structured output from the model."""
        ...


class RateLimitConfig(TypedDict, total=False):
    """Configuration for rate limiting.

    Attributes:
        rpm: Requests per minute limit.
        bucket_key: Optional key for sharing rate limit buckets.
        timeout: Timeout for acquiring tokens in seconds.
        window: Time window for rate calculation in seconds.
    """

    rpm: int
    bucket_key: Optional[str]
    timeout: Optional[float]
    window: Optional[int]


class TokenBucket:
    """Token bucket implementation for rate limiting.

    Implements the token bucket algorithm which allows burst capacity while
    maintaining an average rate limit over time. Tokens are refilled at a
    constant rate up to the bucket capacity.

    This implementation is thread-safe and can be shared across multiple
    consumers.

    Example:
        ```python
        # Create a bucket with 60 requests per minute
        bucket = TokenBucket(capacity=60, window=60)

        # Try to acquire a token
        if bucket.try_acquire():
            # Make API call
            pass
        else:
            # Rate limit exceeded
            pass
        ```
    """

    def __init__(self, capacity: int, window: int = 60) -> None:
        """Initialize the token bucket.

        Args:
            capacity: Maximum number of tokens (requests) in the bucket.
            window: Time window in seconds over which the rate is calculated.
                Defaults to 60 seconds (1 minute).
        """
        self._capacity = capacity
        self._window = window
        self._refill_rate = window / capacity if capacity > 0 else float("inf")  # seconds per token
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

        # Create lock - isolated for easy async conversion
        self._create_lock()

        logger.debug(
            "capacity=<%s>, window=<%s>, refill_rate=<%s> | initialized token bucket",
            capacity,
            window,
            self._refill_rate,
        )

    def _create_lock(self) -> None:
        """Create the synchronization lock."""
        self._lock = threading.Lock()

    def _sleep(self, duration: float) -> None:
        """Sleep for the specified duration.

        Args:
            duration: Time to sleep in seconds.
        """
        time.sleep(duration)

    def _current_time(self) -> float:
        """Get current monotonic time.

        Isolated for easier testing and potential async modifications.

        Returns:
            Current time in seconds.
        """
        return time.monotonic()

    def try_acquire(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Attempt to acquire tokens from the bucket.

        Args:
            count: Number of tokens to acquire.
                Defaults to 1.
            timeout: Maximum time to wait for tokens in seconds.
                If None, blocks indefinitely.

        Returns:
            True if tokens were acquired, False if timeout occurred.

        Raises:
            ValueError: If count exceeds bucket capacity.
        """
        if count > self._capacity:
            raise ValueError(f"Requested tokens ({count}) exceeds capacity ({self._capacity})")

        deadline = self._current_time() + timeout if timeout is not None else float("inf")

        # Use a do-while pattern to ensure we check at least once even with timeout=0
        first_iteration = True

        while first_iteration or self._current_time() < deadline:
            first_iteration = False

            with self._lock:
                self._refill()

                if self._tokens >= count:
                    self._tokens -= count
                    logger.debug(
                        "tokens_acquired=<%s>, tokens_remaining=<%s> | acquired tokens", count, int(self._tokens)
                    )
                    return True

                # Calculate wait time until enough tokens are available
                tokens_needed = count - self._tokens
                wait_time = tokens_needed * self._refill_rate

                # If refill rate is infinite (zero capacity), no point waiting
                if self._refill_rate == float("inf"):
                    break

                if timeout is not None:
                    wait_time = min(wait_time, deadline - self._current_time())
                    if wait_time <= 0:
                        break

            if wait_time > 0:
                logger.debug("wait_time=<%s> | waiting for tokens", wait_time)
                self._sleep(wait_time)

        logger.debug("count=<%s>, timeout=<%s> | token acquisition timed out", count, timeout)
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time.

        Must be called while holding the lock.
        """
        now = self._current_time()
        elapsed = now - self._last_refill

        tokens_to_add = elapsed / self._refill_rate
        self._tokens = min(self._capacity, self._tokens + tokens_to_add)
        self._last_refill = now


class RateLimiterRegistry:
    """Registry for managing shared rate limit buckets.

    This registry ensures that multiple model instances can share the same
    rate limit bucket when configured with the same bucket key. Buckets are
    automatically removed when no longer referenced by any model instances.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._buckets: weakref.WeakValueDictionary[str, TokenBucket] = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    def get_or_create_bucket(self, key: str, capacity: int, window: int = 60) -> TokenBucket:
        """Get an existing bucket or create a new one.

        Args:
            key: Unique identifier for the bucket.
            capacity: Token capacity if creating a new bucket.
            window: Time window if creating a new bucket.

        Returns:
            The token bucket instance.
        """
        with self._lock:
            # Try to get existing bucket
            bucket = self._buckets.get(key)
            if bucket is None:
                logger.debug("bucket_key=<%s>, capacity=<%s> | creating new rate limit bucket", key, capacity)
                bucket = TokenBucket(capacity, window)
                self._buckets[key] = bucket
            return bucket

    def remove_bucket(self, key: str) -> None:
        """Remove a bucket from the registry.

        Args:
            key: The bucket key to remove.
        """
        with self._lock:
            if key in self._buckets:
                logger.debug("bucket_key=<%s> | removing rate limit bucket", key)
                del self._buckets[key]

    def clear(self) -> None:
        """Clear all buckets from the registry.

        Useful for testing or resetting rate limits.
        """
        with self._lock:
            self._buckets.clear()
            logger.debug("cleared all rate limit buckets from registry")


# Global registry instance
_rate_limiter_registry = RateLimiterRegistry()

# Use TypeVar to preserve the original type when wrapping
T = TypeVar("T", bound=RateLimitableModel)


class RateLimitedModel(Generic[T]):
    """Wrapper that adds rate limiting to any model provider.

    This wrapper intercepts calls to model methods and applies rate limiting
    using a token bucket algorithm. Multiple instances can share rate limit
    buckets to coordinate API usage across different agents or components.

    The wrapper is transparent - all attributes and methods not related to
    rate limiting are delegated to the underlying model.

    Example:
        Basic usage with a single agent:
        ```python
        from strands import Agent
        from strands.models import BedrockModel
        from strands.models.rate_limiter import RateLimitedModel, RateLimitConfig

        # Create a rate-limited model
        model = BedrockModel(model_id="us.anthropic.claude-opus-4-20250514-v1:0")
        config = RateLimitConfig(rpm=60, timeout=30.0)
        limited_model = RateLimitedModel(model, config)

        # Use it with an agent
        agent = Agent(model=limited_model)
        response = agent("Hello!")
        ```

        Sharing rate limits across multiple agents:
        ```python
        # Both models share the same rate limit bucket
        config = RateLimitConfig(rpm=60, bucket_key="shared-claude")

        model1 = RateLimitedModel(BedrockModel(...), config)
        model2 = RateLimitedModel(BedrockModel(...), config)

        agent1 = Agent(model=model1)
        agent2 = Agent(model=model2)
        # Both agents share the 60 RPM limit
        ```
    """

    def __init__(
        self,
        model: T,
        config: RateLimitConfig,
    ) -> None:
        """Initialize the rate-limited model wrapper.

        Args:
            model: The model instance to wrap.
            config: Rate limiting configuration.

        Raises:
            ValueError: If rpm is not positive.
        """
        rpm = config.get("rpm", 60)
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")

        self._model = model
        self._config = config

        # Determine bucket key
        bucket_key = config.get("bucket_key")
        if bucket_key is None:
            # Generate key from model type and config
            model_class = type(model).__name__
            model_id = "unknown"

            # Try to get model_id from common patterns
            if hasattr(model, "config"):
                model_config = model.config
                if isinstance(model_config, dict):
                    model_id = model_config.get("model_id", "unknown")
                elif hasattr(model_config, "get"):
                    model_id = model_config.get("model_id", "unknown")

            bucket_key = f"{model_class}:{model_id}:{rpm}"

        # Get or create bucket
        window = config.get("window") or 60  # Use 'or' to ensure int type for mypy
        self._bucket = _rate_limiter_registry.get_or_create_bucket(key=bucket_key, capacity=rpm, window=window)

        logger.info(
            "model_type=<%s>, rpm=<%s>, bucket_key=<%s> | initialized rate-limited model wrapper",
            type(model).__name__,
            rpm,
            bucket_key,
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model.

        Args:
            name: Attribute name.

        Returns:
            The requested attribute from the wrapped model.
        """
        return getattr(self._model, name)

    def _acquire_token(self, method_name: str) -> None:
        """Acquire a rate limit token for a method call.

        Args:
            method_name: Name of the method being called.

        Raises:
            TimeoutError: If token acquisition times out.
        """
        timeout = self._config.get("timeout")

        if not self._bucket.try_acquire(timeout=timeout):
            logger.error("method=<%s>, timeout=<%s> | failed to acquire rate limit token", method_name, timeout)
            raise TimeoutError(f"Rate limit token acquisition timed out after {timeout}s")

        logger.debug("method=<%s> | acquired rate limit token", method_name)

    def converse(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited converse method.

        Args:
            *args: Arguments for the model's converse method.
            **kwargs: Keyword arguments for the model's converse method.

        Returns:
            Result from the model's converse method.

        Raises:
            TimeoutError: If rate limit token acquisition times out.
        """
        self._acquire_token("converse")
        return self._model.converse(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited stream method.

        Args:
            *args: Arguments for the model's stream method.
            **kwargs: Keyword arguments for the model's stream method.

        Returns:
            Result from the model's stream method.

        Raises:
            TimeoutError: If rate limit token acquisition times out.
        """
        self._acquire_token("stream")
        return self._model.stream(*args, **kwargs)

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Rate-limited structured_output method.

        Args:
            *args: Arguments for the model's structured_output method.
            **kwargs: Keyword arguments for the model's structured_output method.

        Returns:
            Result from the model's structured_output method.

        Raises:
            TimeoutError: If rate limit token acquisition times out.
            AttributeError: If the wrapped model doesn't have structured_output.
        """
        if hasattr(self._model, "structured_output"):
            self._acquire_token("structured_output")
            return self._model.structured_output(*args, **kwargs)
        else:
            raise AttributeError(f"{type(self._model).__name__} has no attribute 'structured_output'")


@overload
def rate_limit_model(
    model_or_class: T,
    rpm: int,
    *,
    bucket_key: Optional[str] = None,
    timeout: Optional[float] = None,
    window: int = 60,
) -> RateLimitedModel[T]: ...


@overload
def rate_limit_model(
    model_or_class: Type[T],
    rpm: int,
    *,
    bucket_key: Optional[str] = None,
    timeout: Optional[float] = None,
    window: int = 60,
) -> Type[RateLimitedModel[T]]: ...


def rate_limit_model(
    model_or_class: Union[T, Type[T]],
    rpm: int,
    *,
    bucket_key: Optional[str] = None,
    timeout: Optional[float] = None,
    window: int = 60,
) -> Union[RateLimitedModel[T], Type[RateLimitedModel[T]]]:
    """Apply rate limiting to a model instance or class.

    This function can be used in two ways:
    1. Wrap an existing model instance with rate limiting
    2. Create a rate-limited model class that can be instantiated

    Args:
        model_or_class: Either a Model instance or a Model class.
        rpm: Requests per minute limit.
        bucket_key: Optional key for sharing rate limit buckets.
            When multiple models use the same bucket_key, they share the same rate limit.
        timeout: Timeout for acquiring rate limit tokens in seconds.
            If None, blocks indefinitely waiting for tokens.
        window: Time window for rate calculation in seconds.
            Defaults to 60 seconds.

    Returns:
        If given an instance: RateLimitedModel wrapping the instance.
        If given a class: A new class that creates rate-limited instances.

    Example:
        Wrap an existing model instance:
        ```python
        from strands import Agent
        from strands.models import BedrockModel
        from strands.models.rate_limiter import rate_limit_model

        model = BedrockModel(model_id="us.anthropic.claude-opus-4-20250514-v1:0")
        limited_model = rate_limit_model(model, rpm=60, timeout=30.0)

        agent = Agent(model=limited_model)
        response = agent("Tell me about rate limiting")
        ```

        Create a rate-limited model class:
        ```python
        from strands import Agent
        from strands.models import BedrockModel, OpenAIModel
        from strands.models.rate_limiter import rate_limit_model

        # Create rate-limited classes
        LimitedBedrockModel = rate_limit_model(BedrockModel, rpm=60)
        LimitedOpenAIModel = rate_limit_model(OpenAIModel, rpm=120)

        # Instantiate multiple agents with shared limits
        agent1 = Agent(model=LimitedBedrockModel(model_id="us.anthropic.claude-opus-4-20250514-v1:0"))
        agent2 = Agent(model=LimitedBedrockModel(model_id="us.anthropic.claude-opus-4-20250514-v1:0"))
        # Both agents share the same 60 RPM limit

        agent3 = Agent(model=LimitedOpenAIModel(model_id="gpt-4"))
        # This agent has a separate 120 RPM limit
        ```

        Multi-agent coordination with custom bucket keys:
        ```python
        from concurrent.futures import ThreadPoolExecutor
        from strands import Agent
        from strands.models import BedrockModel
        from strands.models.rate_limiter import rate_limit_model

        # All agents share the same rate limit
        def create_agent(name: str) -> Agent:
            model = BedrockModel(model_id="us.anthropic.claude-opus-4-20250514-v1:0")
            limited = rate_limit_model(
                model,
                rpm=60,
                bucket_key="shared-pool"
            )
            return Agent(model=limited, name=name)

        agents = [create_agent(f"agent-{i}") for i in range(5)]

        # Run agents in parallel - they'll coordinate rate limits
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(agent, "Process this")
                for agent in agents
            ]
            results = [f.result() for f in futures]
        ```

    Raises:
        ValueError: If rpm is not positive.
    """
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")

    config = RateLimitConfig(
        rpm=rpm,
        bucket_key=bucket_key,
        timeout=timeout,
        window=window,
    )

    if isinstance(model_or_class, type):
        # It's a class - create a factory function that looks like a class
        def create_rate_limited(*args: Any, **kwargs: Any) -> RateLimitedModel[T]:
            """Create a rate-limited instance of the model."""
            instance = model_or_class(*args, **kwargs)
            return RateLimitedModel(instance, config)

        # Make it look like a class for better user experience
        create_rate_limited.__name__ = f"RateLimited{model_or_class.__name__}"
        create_rate_limited.__qualname__ = f"RateLimited{model_or_class.__qualname__}"
        create_rate_limited.__module__ = model_or_class.__module__

        # Type-wise, return it as a class type
        return cast(Type[RateLimitedModel[T]], create_rate_limited)
    else:
        # It's an instance - wrap it directly
        return RateLimitedModel(model_or_class, config)


def reset_rate_limits_for_testing() -> None:
    """Reset all rate limit buckets for testing purposes.

    This function clears the global rate limiter registry, removing all
    existing buckets. It should only be used in test environments to ensure
    test isolation.

    Warning:
        This will affect all rate limited models globally. Only use in tests.

    Example:
        ```python
        from strands.models.rate_limiter import reset_rate_limits_for_testing

        def test_something():
            # Clear any existing rate limits from other tests
            reset_rate_limits_for_testing()

            # Your test code here
            model = rate_limit_model(...)
        ```
    """
    _rate_limiter_registry.clear()
    logger.debug("reset rate limits for testing")
