"""This module provides specialized error handlers for common issues that may occur during event loop execution.

Examples include throttling exceptions and context window overflow errors. These handlers implement recovery strategies
like exponential backoff for throttling and message truncation for context window limitations.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from ..telemetry.metrics import EventLoopMetrics
from ..types.content import Message, Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.models import Model
from ..types.streaming import StopReason
from .message_processor import find_last_message_with_tool_results, truncate_tool_results

logger = logging.getLogger(__name__)


def handle_throttling_error(
    e: ModelThrottledException,
    attempt: int,
    max_attempts: int,
    current_delay: int,
    max_delay: int,
    callback_handler: Any,
    kwargs: Dict[str, Any],
) -> Tuple[bool, int]:
    """Handle throttling exceptions from the model provider with exponential backoff.

    Args:
        e: The exception that occurred during model invocation.
        attempt: Number of times event loop has attempted model invocation.
        max_attempts: Maximum number of retry attempts allowed.
        current_delay: Current delay in seconds before retrying.
        max_delay: Maximum delay in seconds (cap for exponential growth).
        callback_handler: Callback for processing events as they happen.
        kwargs: Additional arguments to pass to the callback handler.

    Returns:
        A tuple containing:
            - bool: True if retry should be attempted, False otherwise
            - int: The new delay to use for the next retry attempt
    """
    if attempt < max_attempts - 1:  # Don't sleep on last attempt
        logger.debug(
            "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
            "| throttling exception encountered "
            "| delaying before next retry",
            current_delay,
            max_attempts,
            attempt + 1,
        )
        callback_handler(event_loop_throttled_delay=current_delay, **kwargs)
        time.sleep(current_delay)
        new_delay = min(current_delay * 2, max_delay)  # Double delay each retry
        return True, new_delay

    callback_handler(force_stop=True, force_stop_reason=str(e))
    return False, current_delay


def handle_input_too_long_error(
    e: ContextWindowOverflowException,
    messages: Messages,
    model: Model,
    system_prompt: Optional[str],
    tool_config: Any,
    callback_handler: Any,
    tool_handler: Any,
    kwargs: Dict[str, Any],
) -> Tuple[StopReason, Message, EventLoopMetrics, Any]:
    """Handle 'Input is too long' errors by truncating tool results.

    When a context window overflow exception occurs (input too long for the model), this function attempts to recover
    by finding and truncating the most recent tool results in the conversation history. If truncation is successful, the
    function will make a call to the event loop.

    Args:
        e: The ContextWindowOverflowException that occurred.
        messages: The conversation message history.
        model: Model provider for running inference.
        system_prompt: System prompt for the model.
        tool_config: Tool configuration for the conversation.
        callback_handler: Callback for processing events as they happen.
        tool_handler: Handler for tool execution.
        kwargs: Additional arguments for the event loop.

    Returns:
        The results from the event loop call if successful.

    Raises:
        ContextWindowOverflowException: If messages cannot be truncated.
    """
    from .event_loop import recurse_event_loop  # Import here to avoid circular imports

    # Find the last message with tool results
    last_message_with_tool_results = find_last_message_with_tool_results(messages)

    # If we found a message with toolResult
    if last_message_with_tool_results is not None:
        logger.debug("message_index=<%s> | found message with tool results at index", last_message_with_tool_results)

        # Truncate the tool results in this message
        truncate_tool_results(messages, last_message_with_tool_results)

        return recurse_event_loop(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_config=tool_config,
            callback_handler=callback_handler,
            tool_handler=tool_handler,
            **kwargs,
        )

    # If we can't handle this error, pass it up
    callback_handler(force_stop=True, force_stop_reason=str(e))
    logger.error("an exception occurred in event_loop_cycle | %s", e)
    raise ContextWindowOverflowException() from e
