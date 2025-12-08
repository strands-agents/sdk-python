"""Custom bidi exceptions."""
from typing import Any


class BidiExceptionChain(Exception):
    """Chain a list of exceptions together.

    Useful for chaining together exceptions raised across a multi-step workflow (e.g., the bidi `stop` methods).
    Note, this exception is meant to mimic ExceptionGroup released in Python 3.11.

    - Docs: https://docs.python.org/3/library/exceptions.html#ExceptionGroup
    """

    def __init__(self, message: str, exceptions: list[Exception]) -> None:
        """Chain exceptions.

        Args:
            message: Top-level exception message.
            exceptions: List of exceptions to chain.
        """
        super().__init__(self, message)

        exceptions.append(self)
        for i in range(1, len(exceptions)):
            exceptions[i].__context__ = exceptions[i - 1]


class BidiModelTimeoutError(Exception):
    """Model timeout error.

    Bidirectional models are often configured with a connection time limit. Nova sonic for example keeps the connection
    open for 8 minutes max. Upon receiving a timeout, the agent loop is configured to restart the model connection so as
    to create a seamless, uninterrupted experience for the user.
    """

    def __init__(self, message: str, **restart_config: Any) -> None:
        """Initialize error.

        Args:
            message: Timeout message from model.
            **restart_config: Configure restart specific behaviors in the call to model start.
        """
        super().__init__(self, message)

        self.restart_config = restart_config
