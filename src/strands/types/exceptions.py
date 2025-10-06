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


class MaxTokensReachedException(Exception):
    """Exception raised when the model reaches its maximum token generation limit.

    This exception is raised when the model stops generating tokens because it has reached the maximum number of
    tokens allowed for output generation. This can occur when the model's max_tokens parameter is set too low for
    the complexity of the response, or when the model naturally reaches its configured output limit during generation.
    """

    def __init__(self, message: str):
        """Initialize the exception with an error message and the incomplete message object.

        Args:
            message: The error message describing the token limit issue
        """
        super().__init__(message)


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


class SessionException(Exception):
    """Exception raised when session operations fail."""

    pass
