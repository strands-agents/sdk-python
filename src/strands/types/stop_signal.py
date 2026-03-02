"""Thread-safe stop signal for graceful agent execution cancellation.

This module provides a simple, thread-safe mechanism for signaling cancellation
to running agent executions. The signal is internal to the Agent class and is
checked at various checkpoints during agent execution.
"""

import threading


class StopSignal:
    """Thread-safe signal for stopping agent execution.

    This class provides a simple mechanism for gracefully stopping agent
    execution. It is used internally by the Agent class and accessed via
    the agent.cancel() method.

    The signal uses a threading lock to ensure thread-safe access to the
    cancellation state, allowing cancellation from any thread or async context.

    Design principles:
    - Thread-safe: Can be cancelled from any thread
    - Lightweight: Minimal overhead for checking cancellation state
    - Simple: Provides cancel() method and internal is_cancelled() check

    Note:
        This class is internal to the Agent implementation. Users should
        call agent.cancel() instead of creating StopSignal instances directly.
    """

    def __init__(self) -> None:
        """Initialize a new stop signal."""
        self._stopped = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal cancellation request.

        This method is thread-safe and can be called from any thread.
        Multiple calls to cancel() are safe and idempotent.
        """
        with self._lock:
            self._stopped = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        This method is thread-safe and can be called from any thread.

        Returns:
            True if cancellation has been requested, False otherwise.
        """
        with self._lock:
            return self._stopped
