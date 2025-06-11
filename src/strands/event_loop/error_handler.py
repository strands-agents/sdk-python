"""This module provides specialized error handlers for common issues that may occur during event loop execution.

Examples include throttling exceptions and context window overflow errors. These handlers implement recovery strategies
like exponential backoff for throttling and message truncation for context window limitations.
"""

import logging
import time
from typing import Any, Dict, Tuple

from ..types.exceptions import ModelThrottledException

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
