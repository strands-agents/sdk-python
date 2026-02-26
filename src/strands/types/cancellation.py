"""Cancellation token types for graceful agent termination."""

import threading


class CancellationToken:
    """Thread-safe cancellation token for graceful agent termination.

    This token can be used to signal cancellation requests from any thread
    and checked synchronously during agent execution. When cancelled, the
    agent will stop processing and yield a stop event with interrupt reasoning.

    Example:
        ```python
        token = CancellationToken()

        # In another thread or external system
        token.cancel()

        # In agent execution
        if token.is_cancelled():
            # Stop processing
            pass
        ```

    Note:
        This is a minimal implementation focused on cancellation signaling.
        Callback registration for resource cleanup can be added in a future
        phase if resource cleanup use cases emerge.
    """

    def __init__(self) -> None:
        """Initialize a new cancellation token."""
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal cancellation request.

        This method is thread-safe and can be called from any thread.
        Multiple calls to cancel() are safe and idempotent.
        """
        with self._lock:
            self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        This method is thread-safe and can be called from any thread.

        Returns:
            True if cancellation has been requested, False otherwise.
        """
        with self._lock:
            return self._cancelled
