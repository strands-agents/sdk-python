"""Exception-related type definitions for the SDK."""

from typing import Any


class EventLoopException(Exception):
    """Exception raised by the event loop."""

    def __init__(self, original_exception: Exception, request_state: Any = None) -> None:
        """Initialize exception.

        Args:
            original_exception: The original exception that was raised.
            request_state: The state of the request at the time of the exception.
        """
        self.original_exception = original_exception
        self.request_state = request_state if request_state is not None else {}
        super().__init__(str(original_exception))


class ContextWindowOverflowException(Exception):
    """Exception raised when the context window is exceeded.

    This exception is raised when the input to a model exceeds the maximum context window size that the model can
    handle. This typically occurs when the combined length of the conversation history, system prompt, and current
    message is too large for the model to process.
    """

    pass


class MCPClientInitializationError(Exception):
    """Raised when the MCP server fails to initialize properly."""

    pass


class ModelThrottledException(Exception):
    """Exception raised when the model is throttled.

    This exception is raised when the model is throttled by the service. This typically occurs when the service is
    throttling the requests from the client.
    """

    def __init__(self, message: str) -> None:
        """Initialize exception.

        Args:
            message: The message from the service that describes the throttling.
        """
        self.message = message
        super().__init__(message)

    pass


class ModelAuthenticationException(Exception):
    """Exception raised when model authentication fails.

    This exception is raised when the API key or other authentication
    credentials are invalid or expired.
    """

    pass


class ModelValidationException(Exception):
    """Exception raised when model input validation fails.

    This exception is raised when the input parameters don't meet the
    model's requirements (e.g., invalid formats, out-of-range values).
    """

    pass


class ContentModerationException(Exception):
    """Exception raised when content is flagged by safety filters.

    This exception is raised when the model's safety systems reject
    the input or output content as inappropriate.
    """

    pass


class ModelServiceException(Exception):
    """Exception raised for model service errors.

    This is a general exception for server-side errors that aren't
    covered by more specific exceptions.
    """

    def __init__(self, message: str, is_transient: bool = False) -> None:
        """Initialize exception.

        Args:
            message: Error message
            is_transient: Whether the error is likely transient (retryable)
        """
        self.message = message
        self.is_transient = is_transient
        super().__init__(message)
