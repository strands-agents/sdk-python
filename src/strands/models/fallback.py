"""FallbackModel implementation for automatic failover between models.

This module provides the FallbackModel class, which wraps two Model instances (primary and fallback)
and automatically switches to the fallback model when the primary model fails with retryable errors.
The implementation is provider-agnostic and works with any combination of Strands model types.

Key Features:
- **Automatic Failover**: Switches to fallback on throttling, connection, and network errors
- **Circuit Breaker Pattern**: Temporarily skips failing primary model to prevent cascading failures
- **Provider Agnostic**: Works with any combination of model providers (OpenAI→Bedrock, etc.)
- **Configurable Behavior**: Customizable thresholds, error detection, and statistics tracking
- **Full Model Interface**: Supports both streaming and structured output methods
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

Circuit Breaker Behavior:
The circuit breaker monitors primary model failures within a sliding time window. When the failure
threshold is exceeded, it "opens" and routes all requests directly to the fallback model for a
cooldown period. This prevents wasting time and resources on a consistently failing primary model.

Circuit States:
- **Closed** (default): Attempt primary model for each request
- **Open**: Skip primary model, use fallback model directly
- **Half-Open**: After cooldown, next request tests if primary has recovered

Error Classification:
By default, the following errors trigger fallback:
- ModelThrottledException (rate limiting)
- Connection errors (network, timeout, refused, unavailable, etc.)
- Custom errors via the should_fallback configuration function

Non-retryable errors (like ContextWindowOverflowException) are re-raised without fallback.

Example usage:
    ```python
    from strands.models import FallbackModel, BedrockModel, OpenAIModel

    # Same-provider fallback (different model sizes)
    model = FallbackModel(
        primary=BedrockModel(model_id="claude-3-opus"),
        fallback=BedrockModel(model_id="claude-3-haiku"),
        circuit_failure_threshold=3,
        circuit_time_window=60.0,
        circuit_cooldown_seconds=30
    )

    # Cross-provider fallback
    cross_provider = FallbackModel(
        primary=OpenAIModel(model_id="gpt-4"),
        fallback=BedrockModel(model_id="claude-3-sonnet"),
        circuit_failure_threshold=5,
        circuit_time_window=120.0
    )

    # Use with an agent
    from strands.agent import Agent
    agent = Agent(model=model)
    response = agent.run("Hello!")

    # Monitor fallback statistics
    stats = model.get_stats()
    print(f"Fallback count: {stats['fallback_count']}")
    print(f"Circuit open: {stats['circuit_open']}")
    ```

"""

import logging
import time
from collections import deque
from typing import Any, AsyncGenerator, Callable, Optional, Type, TypeVar, Union

from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class FallbackConfig(TypedDict, total=False):
    """Configuration for FallbackModel.

    This TypedDict defines the optional configuration parameters for controlling
    the FallbackModel's behavior, including circuit breaker settings and error
    detection logic.

    Attributes:
        circuit_failure_threshold: Number of primary model failures within the time
            window before the circuit breaker opens. Once opened, the primary model
            will be skipped until the cooldown period expires. Default: 3
        circuit_time_window: Time window in seconds for counting failures. Only
            failures within this window are counted toward the threshold. Default: 60.0
        circuit_cooldown_seconds: How long to wait in seconds before retrying the
            primary model after the circuit breaker opens. Default: 30
        should_fallback: Optional custom function that takes an Exception and returns
            a boolean indicating whether to attempt fallback. If provided, this function
            overrides the default error classification logic. Default: None
        track_stats: Whether to track statistics about fallback usage, including
            fallback count, primary failures, and circuit breaker state. Default: True
    """

    circuit_failure_threshold: int
    circuit_time_window: float
    circuit_cooldown_seconds: int
    should_fallback: Optional[Callable[[Exception], bool]]
    track_stats: bool


class FallbackStats(TypedDict):
    """Typed structure for fallback statistics to provide better type hints.

    This TypedDict defines the structure returned by get_stats() method, providing
    comprehensive information about the FallbackModel's current state and usage
    statistics for monitoring and debugging purposes.

    Attributes:
        fallback_count: Total number of times the fallback model was used due to
            primary model failures
        primary_failures: Total number of primary model failures encountered
        circuit_skips: Number of times requests were routed directly to fallback
            because the circuit breaker was open
        using_fallback: Boolean indicating whether the last request used the fallback
            model (True) or primary model (False)
        circuit_open: Boolean indicating whether the circuit breaker is currently
            open, meaning primary model will be skipped
        recent_failures: Number of primary model failures within the current time
            window that count toward opening the circuit breaker
        circuit_open_until: Timestamp (float) when the circuit breaker will close
            and primary model will be retried, or None if circuit is closed
        primary_model_name: Human-readable name/identifier of the primary model
            for debugging and monitoring purposes
        fallback_model_name: Human-readable name/identifier of the fallback model
            for debugging and monitoring purposes
    """

    fallback_count: int
    primary_failures: int
    circuit_skips: int
    using_fallback: bool
    circuit_open: bool
    recent_failures: int
    circuit_open_until: Optional[float]
    primary_model_name: str
    fallback_model_name: str


class FallbackModel(Model):
    """A model that automatically falls back to a secondary model on primary model failures.

    FallbackModel wraps two Model instances (primary and fallback) and provides automatic
    failover when the primary model encounters retryable errors such as throttling,
    connection issues, or network problems.

    FallbackModel implements a circuit breaker pattern to prevent repeated attempts
    to a failing primary model. The circuit breaker opens after a configurable number
    of failures within a time window, temporarily routing all requests to the fallback
    model until the cooldown period expires.

    Example:
        ```python
        from strands.models import FallbackModel, BedrockModel

        # Create a fallback model with two Bedrock models
        model = FallbackModel(
            primary=BedrockModel(model_id="claude-3-opus"),
            fallback=BedrockModel(model_id="claude-3-haiku"),
            circuit_failure_threshold=3,
            circuit_time_window=60.0,
            circuit_cooldown_seconds=30
        )

        # Use with an agent
        agent = Agent(model=model)
        response = agent.run("Hello!")
        ```

    Attributes:
        primary: The primary Model instance to use for requests
        fallback: The fallback Model instance to use when primary fails
        circuit_failure_threshold: Number of failures before circuit opens
        circuit_time_window: Time window in seconds for counting failures
        circuit_cooldown_seconds: Cooldown period before retrying primary
        should_fallback: Optional custom function for error classification
        track_stats: Whether to track usage statistics
    """

    def __init__(
        self,
        *,
        primary: Model,
        fallback: Model,
        **config: Unpack[FallbackConfig],
    ) -> None:
        """Initialize the FallbackModel with primary and fallback models.

        Args:
            primary: The primary Model instance to use for requests
            fallback: The fallback Model instance to use when primary fails
            **config: Configuration options from FallbackConfig TypedDict:
                - circuit_failure_threshold: Number of failures before circuit opens (default: 3)
                - circuit_time_window: Time window in seconds for counting failures (default: 60.0)
                - circuit_cooldown_seconds: Cooldown period in seconds (default: 30)
                - should_fallback: Optional custom error classification function (default: None)
                - track_stats: Whether to track statistics (default: True)
        """
        # Store model instances
        self.primary = primary
        self.fallback = fallback

        # Initialize config with defaults
        self.circuit_failure_threshold = config.get("circuit_failure_threshold", 3)
        self.circuit_time_window = config.get("circuit_time_window", 60.0)
        self.circuit_cooldown_seconds = config.get("circuit_cooldown_seconds", 30)
        self.should_fallback = config.get("should_fallback", None)
        self.track_stats = config.get("track_stats", True)

        # Initialize circuit breaker state
        self._failure_timestamps: deque[float] = deque(maxlen=100)
        self._circuit_open = False
        self._circuit_open_until: Optional[float] = None

        # Initialize statistics
        self._stats: dict[str, Union[int, bool]] = {
            "fallback_count": 0,
            "primary_failures": 0,
            "circuit_skips": 0,
            "using_fallback": False,
        }

        logger.info(
            "primary=<%s>, fallback=<%s>, circuit_failure_threshold=<%d>, "
            "circuit_time_window=<%s>s, circuit_cooldown_seconds=<%d>s | initialized FallbackModel",
            self._get_model_name(primary),
            self._get_model_name(fallback),
            self.circuit_failure_threshold,
            self.circuit_time_window,
            self.circuit_cooldown_seconds,
        )

    def _check_circuit(self) -> bool:
        """Check if the circuit breaker is open and handle cooldown expiration.

        This method checks the current state of the circuit breaker. If the circuit
        is open and the cooldown period has expired, it automatically closes the
        circuit and logs the event.

        Returns:
            True if the circuit is open (primary should be skipped), False if closed
            (primary can be attempted).
        """
        current_time = time.time()

        # Check if circuit is open
        if self._circuit_open:
            # Check if cooldown has expired
            if self._circuit_open_until is not None and current_time >= self._circuit_open_until:
                # Close the circuit
                self._circuit_open = False
                self._circuit_open_until = None
                logger.info("Circuit breaker closed, will retry primary model")
                return False

            # Circuit is still open
            return True

        # Circuit is closed
        return False

    def _handle_primary_failure(self, error: Exception) -> None:
        """Handle a primary model failure and potentially open the circuit breaker.

        This method records the failure timestamp, updates statistics, and checks
        if the circuit breaker threshold has been reached. If the threshold is
        exceeded, the circuit breaker opens and remains open for the configured
        cooldown period.

        Args:
            error: The exception that caused the primary model to fail.
        """
        current_time = time.time()

        # Record the failure timestamp
        self._failure_timestamps.append(current_time)

        # Increment failure counter
        if self.track_stats:
            self._stats["primary_failures"] += 1

        # Count recent failures within the time window
        recent_failures = sum(
            1 for timestamp in self._failure_timestamps if current_time - timestamp <= self.circuit_time_window
        )

        # Check if we should open the circuit
        if recent_failures >= self.circuit_failure_threshold:
            self._circuit_open = True
            self._circuit_open_until = current_time + self.circuit_cooldown_seconds
            logger.warning(
                "recent_failures=<%d>, time_window=<%s>s, cooldown=<%d>s | circuit breaker opened",
                recent_failures,
                self.circuit_time_window,
                self.circuit_cooldown_seconds,
            )

    def _should_fallback(self, error: Exception) -> bool:
        """Determine if an error should trigger fallback to the secondary model.

        This method classifies errors to determine if they are retryable with a
        fallback model. By default, it triggers fallback for throttling and
        connection/network errors, but not for context window overflow errors.

        A custom classification function can be provided via the should_fallback
        configuration parameter to override the default logic.

        Args:
            error: The exception raised by the primary model.

        Returns:
            True if the error should trigger fallback, False otherwise.
        """
        # Check if custom should_fallback function exists in config
        if self.should_fallback is not None:
            return self.should_fallback(error)

        # Check if error is instance of ModelThrottledException
        if isinstance(error, ModelThrottledException):
            return True

        # Convert error to lowercase string
        error_str = str(error).lower()

        # Check if any connection error keywords present in error string
        connection_keywords = [
            "connection",
            "network",
            "timeout",
            "refused",
            "unavailable",
            "closed",
            "aborted",
            "reset",
        ]
        if any(keyword in error_str for keyword in connection_keywords):
            return True

        # Check if error is instance of ContextWindowOverflowException
        if isinstance(error, ContextWindowOverflowException):
            return False

        # Return False as default
        return False

    def _get_model_name(self, model: Model) -> str:
        """Extract a human-readable name for the model for debugging purposes.

        This method attempts to extract a meaningful identifier from the model's
        configuration, falling back to the class name if no specific identifier
        is found. This helps with debugging and monitoring by providing clear
        model identification in statistics and logs.

        Args:
            model: The Model instance to extract a name from.

        Returns:
            A string identifier for the model, either from configuration or class name.
        """
        try:
            # Try to get from model configuration first
            config = model.get_config()
            if isinstance(config, dict):
                # Look for common model identifier fields
                for key in ["model_id", "model", "name"]:
                    if key in config and config[key]:
                        return str(config[key])
        except Exception:
            # If get_config() fails, fall back to class name
            pass

        # Fall back to class name
        return model.__class__.__name__

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the model, with automatic fallback on primary failure.

        This method attempts to stream from the primary model first. If the primary model
        fails with a retryable error (throttling, connection issues), it automatically
        falls back to the secondary model. The circuit breaker may skip the primary
        model entirely if it has failed repeatedly.

        The method handles the following scenarios:
        1. Circuit breaker open: Skip primary and use fallback directly
        2. Primary success: Stream from primary model
        3. Primary failure (retryable): Fall back to secondary model
        4. Primary failure (non-retryable): Re-raise the exception

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments passed to the underlying model.

        Yields:
            StreamEvent objects from either the primary or fallback model.

        Raises:
            Exception: Re-raises non-retryable exceptions from the primary model,
                or exceptions from the fallback model if both models fail.

        Example:
            ```python
            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            async for event in model.stream(messages=[{"role": "user", "content": "Hello"}]):
                print(event)
            ```
        """
        # Check if circuit breaker is open
        if self._check_circuit():
            # Circuit is open, skip primary and use fallback directly
            if self.track_stats:
                self._stats["circuit_skips"] += 1
                self._stats["using_fallback"] = True

            logger.info(
                "fallback_model=<%s> | circuit breaker is open, using fallback model directly",
                self._get_model_name(self.fallback),
            )

            # Yield events from fallback model
            async for event in self.fallback.stream(
                messages=messages,
                tool_specs=tool_specs,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                **kwargs,
            ):
                yield event

            return

        # Circuit is closed, try primary model
        if self.track_stats:
            self._stats["using_fallback"] = False

        try:
            # Attempt to stream from primary model
            async for event in self.primary.stream(
                messages=messages,
                tool_specs=tool_specs,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                **kwargs,
            ):
                yield event

            # Primary model succeeded
            return

        except Exception as error:
            # Primary model failed
            error_message = str(error)[:200]  # Truncate to first 200 chars
            logger.warning(
                "primary_model=<%s>, error_type=<%s>, error_message=<%s> | primary model failed",
                self._get_model_name(self.primary),
                error.__class__.__name__,
                error_message,
            )

            # Check if we should fallback for this error
            if not self._should_fallback(error):
                logger.debug("error_type=<%s> | error is not fallback-eligible, re-raising", error.__class__.__name__)
                raise

            # Error is fallback-eligible, handle the failure
            self._handle_primary_failure(error)

            if self.track_stats:
                self._stats["fallback_count"] += 1
                self._stats["using_fallback"] = True

            logger.info(
                "fallback_model=<%s>, fallback_count=<%d> | attempting fallback",
                self._get_model_name(self.fallback),
                self._stats["fallback_count"],
            )

            try:
                # Attempt to stream from fallback model
                async for event in self.fallback.stream(
                    messages=messages,
                    tool_specs=tool_specs,
                    system_prompt=system_prompt,
                    tool_choice=tool_choice,
                    **kwargs,
                ):
                    yield event

                logger.info("fallback_model=<%s> | fallback succeeded", self._get_model_name(self.fallback))

            except Exception as fallback_error:
                # Both models failed
                logger.error(
                    "primary_model=<%s>, primary_error=<%s>, fallback_model=<%s>, fallback_error=<%s> | "
                    "both models failed",
                    self._get_model_name(self.primary),
                    error.__class__.__name__,
                    self._get_model_name(self.fallback),
                    fallback_error.__class__.__name__,
                )
                # Raise the fallback exception, not the primary
                raise fallback_error

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Generate structured output with the model, with automatic fallback on primary failure.

        This method attempts to generate structured output from the primary model first. If the
        primary model fails with a retryable error (throttling, connection issues), it automatically
        falls back to the secondary model. The circuit breaker may skip the primary model entirely
        if it has failed repeatedly.

        The method handles the following scenarios:
        1. Circuit breaker open: Skip primary and use fallback directly
        2. Primary success: Generate structured output from primary model
        3. Primary failure (retryable): Fall back to secondary model
        4. Primary failure (non-retryable): Re-raise the exception

        Args:
            output_model: Pydantic model class defining the expected output structure.
            prompt: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments passed to the underlying model.

        Yields:
            Dictionary events from either the primary or fallback model, containing
            structured output data conforming to the output_model schema.

        Raises:
            Exception: Re-raises non-retryable exceptions from the primary model,
                or exceptions from the fallback model if both models fail.

        Example:
            ```python
            from pydantic import BaseModel

            class Response(BaseModel):
                answer: str
                confidence: float

            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            async for event in model.structured_output(
                output_model=Response,
                prompt=[{"role": "user", "content": "What is 2+2?"}]
            ):
                if event.get("chunk_type") == "structured_output":
                    print(event["data"])
            ```
        """
        # Check if circuit breaker is open
        if self._check_circuit():
            # Circuit is open, skip primary and use fallback directly
            if self.track_stats:
                self._stats["circuit_skips"] += 1
                self._stats["using_fallback"] = True

            logger.info(
                "fallback_model=<%s> | circuit breaker is open, using fallback model directly for structured output",
                self._get_model_name(self.fallback),
            )

            # Yield events from fallback model
            async for event in self.fallback.structured_output(
                output_model=output_model,
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            ):
                yield event

            return

        # Circuit is closed, try primary model
        if self.track_stats:
            self._stats["using_fallback"] = False

        try:
            # Attempt to get structured output from primary model
            async for event in self.primary.structured_output(
                output_model=output_model,
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            ):
                yield event

            # Primary model succeeded
            return

        except Exception as error:
            # Primary model failed
            error_message = str(error)[:200]  # Truncate to first 200 chars
            logger.warning(
                "primary_model=<%s>, error_type=<%s>, error_message=<%s> | "
                "primary model failed during structured output",
                self._get_model_name(self.primary),
                error.__class__.__name__,
                error_message,
            )

            # Check if we should fallback for this error
            if not self._should_fallback(error):
                logger.debug("error_type=<%s> | error is not fallback-eligible, re-raising", error.__class__.__name__)
                raise

            # Error is fallback-eligible, handle the failure
            self._handle_primary_failure(error)

            if self.track_stats:
                self._stats["fallback_count"] += 1
                self._stats["using_fallback"] = True

            logger.info(
                "fallback_model=<%s>, fallback_count=<%d> | attempting fallback for structured output",
                self._get_model_name(self.fallback),
                self._stats["fallback_count"],
            )

            try:
                # Attempt to get structured output from fallback model
                async for event in self.fallback.structured_output(
                    output_model=output_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs,
                ):
                    yield event

                logger.info(
                    "fallback_model=<%s> | fallback succeeded for structured output",
                    self._get_model_name(self.fallback),
                )

            except Exception as fallback_error:
                # Both models failed
                logger.error(
                    "primary_model=<%s>, primary_error=<%s>, fallback_model=<%s>, fallback_error=<%s> | "
                    "both models failed during structured output",
                    self._get_model_name(self.primary),
                    error.__class__.__name__,
                    self._get_model_name(self.fallback),
                    fallback_error.__class__.__name__,
                )
                # Raise the fallback exception, not the primary
                raise fallback_error

    @override
    def update_config(self, **config: Any) -> None:
        """Update the FallbackModel configuration.

        This method updates the configuration parameters for the FallbackModel itself,
        such as circuit breaker thresholds and error detection logic. It does not
        affect the configuration of the underlying primary or fallback models.

        Args:
            **config: Configuration options from FallbackConfig TypedDict:
                - circuit_failure_threshold: Number of failures before circuit opens
                - circuit_time_window: Time window in seconds for counting failures
                - circuit_cooldown_seconds: Cooldown period in seconds
                - should_fallback: Optional custom error classification function
                - track_stats: Whether to track statistics

        Example:
            ```python
            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            # Update circuit breaker thresholds
            model.update_config(
                circuit_failure_threshold=5,
                circuit_time_window=120.0
            )
            ```

        Note:
            This method only updates the FallbackModel's configuration. To update
            the configuration of the underlying primary or fallback models, call
            their respective update_config() methods directly.
        """
        # Update circuit_failure_threshold if provided
        if "circuit_failure_threshold" in config:
            self.circuit_failure_threshold = config["circuit_failure_threshold"]

        # Update circuit_time_window if provided
        if "circuit_time_window" in config:
            self.circuit_time_window = config["circuit_time_window"]

        # Update circuit_cooldown_seconds if provided
        if "circuit_cooldown_seconds" in config:
            self.circuit_cooldown_seconds = config["circuit_cooldown_seconds"]

        # Update should_fallback if provided
        if "should_fallback" in config:
            self.should_fallback = config["should_fallback"]

        # Update track_stats if provided
        if "track_stats" in config:
            self.track_stats = config["track_stats"]

    def get_stats(self) -> FallbackStats:
        """Get current statistics about fallback usage and circuit breaker state.

        This method returns comprehensive statistics including fallback counts,
        primary failures, circuit breaker state, recent failure counts within
        the configured time window, and model identifiers for debugging.

        Returns:
            FallbackStats containing:
                - fallback_count: Total number of times fallback was used
                - primary_failures: Total number of primary model failures
                - circuit_skips: Number of times circuit breaker skipped primary
                - using_fallback: Whether currently using fallback model
                - circuit_open: Current circuit breaker state (True if open)
                - recent_failures: Number of failures within the time window
                - circuit_open_until: Timestamp when circuit will close (or None)
                - primary_model_name: Name/identifier of the primary model
                - fallback_model_name: Name/identifier of the fallback model

        Example:
            ```python
            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            # After some usage
            stats = model.get_stats()
            print(f"Fallback count: {stats['fallback_count']}")
            print(f"Circuit open: {stats['circuit_open']}")
            print(f"Primary model: {stats['primary_model_name']}")
            print(f"Fallback model: {stats['fallback_model_name']}")
            ```
        """
        # Get current timestamp
        current_time = time.time()

        # Count recent failures within time window using list comprehension
        recent_failures = sum(
            1 for timestamp in self._failure_timestamps if current_time - timestamp <= self.circuit_time_window
        )

        # Return FallbackStats with all required fields
        return FallbackStats(
            fallback_count=int(self._stats["fallback_count"]),
            primary_failures=int(self._stats["primary_failures"]),
            circuit_skips=int(self._stats["circuit_skips"]),
            using_fallback=bool(self._stats["using_fallback"]),
            circuit_open=self._circuit_open,
            recent_failures=recent_failures,
            circuit_open_until=self._circuit_open_until,
            primary_model_name=self._get_model_name(self.primary),
            fallback_model_name=self._get_model_name(self.fallback),
        )

    def reset_stats(self) -> None:
        """Reset all statistics and circuit breaker state.

        This method clears all tracked statistics, failure timestamps, and resets
        the circuit breaker to its initial closed state. This is useful for testing
        or when you want to start fresh after resolving issues with the primary model.

        Example:
            ```python
            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            # After some usage and failures
            model.reset_stats()

            # Statistics are now cleared
            stats = model.get_stats()
            assert stats['fallback_count'] == 0
            assert stats['circuit_open'] == False
            ```

        Note:
            This method resets the FallbackModel's statistics and circuit breaker
            state, but does not affect the underlying primary or fallback models.
        """
        # Clear _failure_timestamps deque
        self._failure_timestamps.clear()

        # Set _circuit_open=False and _circuit_open_until=None
        self._circuit_open = False
        self._circuit_open_until = None

        # Reset _stats dict to initial values
        self._stats = {
            "fallback_count": 0,
            "primary_failures": 0,
            "circuit_skips": 0,
            "using_fallback": False,
        }

    @override
    def get_config(self) -> dict[str, Any]:
        """Get the complete configuration including FallbackModel and underlying models.

        This method returns a comprehensive view of the configuration, including:
        - FallbackModel's own configuration (circuit breaker settings, etc.)
        - Primary model's configuration
        - Fallback model's configuration
        - Current statistics (if tracking is enabled)

        Returns:
            Dictionary with keys:
                - fallback_config: FallbackModel configuration parameters
                - primary_config: Configuration from the primary model
                - fallback_model_config: Configuration from the fallback model
                - stats: Current statistics (if track_stats is True, otherwise None)

        Example:
            ```python
            model = FallbackModel(primary=primary_model, fallback=fallback_model)

            config = model.get_config()
            print(f"Circuit threshold: {config['fallback_config']['circuit_failure_threshold']}")
            print(f"Fallback count: {config['stats']['fallback_count']}")
            ```
        """
        # Build fallback_config dictionary
        fallback_config = {
            "circuit_failure_threshold": self.circuit_failure_threshold,
            "circuit_time_window": self.circuit_time_window,
            "circuit_cooldown_seconds": self.circuit_cooldown_seconds,
            "should_fallback": self.should_fallback,
            "track_stats": self.track_stats,
        }

        # Get primary model config
        primary_config = self.primary.get_config()

        # Get fallback model config
        fallback_model_config = self.fallback.get_config()

        # Get stats if tracking is enabled
        stats = self.get_stats() if self.track_stats else None

        return {
            "fallback_config": fallback_config,
            "primary_config": primary_config,
            "fallback_model_config": fallback_model_config,
            "stats": stats,
        }
