"""Abstract interface for conversation history management."""

from abc import ABC, abstractmethod
from typing import Optional

from ...types.content import Messages


class ConversationManager(ABC):
    """Abstract base class for managing conversation history.

    This class provides an interface for implementing conversation management strategies to control the size of message
    arrays/conversation histories, helping to:

    - Manage memory usage
    - Control context length
    - Maintain relevant conversation state
    """

    @abstractmethod
    # pragma: no cover
    def apply_management(self, messages: Messages) -> None:
        """Applies management strategy to the provided list of messages.

        Processes the conversation history to maintain appropriate size by modifying the messages list in-place.
        Implementations should handle message pruning, summarization, or other size management techniques to keep the
        conversation context within desired bounds.

        Args:
            messages: The conversation history to manage.
                This list is modified in-place.
        """
        pass

    @abstractmethod
    # pragma: no cover
    def reduce_context(self, messages: Messages, e: Optional[Exception] = None) -> None:
        """Called when the model's context window is exceeded.

        This method should implement the specific strategy for reducing the window size when a context overflow occurs.
        It is typically called after a ContextWindowOverflowException is caught.

        Implementations might use strategies such as:

        - Removing the N oldest messages
        - Summarizing older context
        - Applying importance-based filtering
        - Maintaining critical conversation markers

        Args:
            messages: The conversation history to reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.
        """
        pass
