"""Exception classes for session management operations."""

from typing import Optional


class SessionException(Exception):
    """Exception raised when session operations fail.

    This exception is raised for various session-related failures including:
    - Session creation failures
    - Session read/write failures (corrupted data, missing sessions)
    - Session deletion failures
    - Session listing failures

    The exception wraps underlying errors while providing context about the
    session operation that failed.
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """Initialize the session exception.

        Args:
            message: Human-readable description of the session failure
            cause: Optional underlying exception that caused the session failure
        """
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.cause:
            return f"{super().__str__()} (caused by: {self.cause})"
        return super().__str__()
