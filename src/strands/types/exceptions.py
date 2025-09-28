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

class AgentDelegationException(Exception):
    """ Exception raised when an agent delegates to a sub-agent. 
    This exception provides a clean control flow mechanism for agent delegation, 
    allowing immediate termination of the orchestrator and transfer of exectuion
    to the specific sub-agent.
    Note: using exception for control flow is intentional here as it provides a clean
    way to short-circuit the event loop without refactoring the entire execution pipeline.
    while exceptions are typically for errors, in this case we use it in a similiar usecase
    as if we were using it in stopIteran in generators - it's a strutured way to signal
    completion of a specidic control fl;ow path.
    For delegation opnerations(which are not high frequency in nature) this approach
    maintains simplicity and avoids complex return value handling throughout the tool execution stack
    """
    def __init__(
        self,
        target_agent:str,
        message:str,
        context:dict[str,Any] | None = None,
        delegation_chain:list[str] | None = None,
        transfer_state: bool= True,
        transfer_messages: bool= True, 
    )-> None:
        self.target_agent=target_agent
        self.message=message
        self.context=context or {}
        self.delegation_chain= delegation_chain or []
        self.transfer_state=transfer_state
        self.transfer_message=transfer_messages
        super().__init__(f"Delgating to an Agent: {target_agent}")
        