"""Abstract interface for conversation history management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ...hooks.registry import HookProvider, HookRegistry
from ...types.content import Message

if TYPE_CHECKING:
    from ...agent.agent import Agent


class ConversationManager(ABC, HookProvider):
    """Abstract base class for managing conversation history.

    This class provides an interface for implementing conversation management strategies to control the size of message
    arrays/conversation histories, helping to:

    - Manage memory usage
    - Control context length
    - Maintain relevant conversation state

    ConversationManager implements the HookProvider protocol, allowing derived classes to register hooks for agent
    lifecycle events. Derived classes that override register_hooks must call the base implementation to ensure proper
    hook registration.

    Example:
        ```python
        class MyConversationManager(ConversationManager):
            def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
                super().register_hooks(registry, **kwargs)
                # Register additional hooks here
        ```
    """

    def __init__(self) -> None:
        """Initialize the ConversationManager.

        Attributes:
          removed_message_count: The messages that have been removed from the agents messages array.
              These represent messages provided by the user or LLM that have been removed, not messages
              included by the conversation manager through something like summarization.
        """
        self.removed_message_count = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for agent lifecycle events.

        Derived classes that override this method must call the base implementation to ensure proper hook
        registration chain.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.

        Example:
            ```python
            def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
                super().register_hooks(registry, **kwargs)
                registry.add_callback(SomeEvent, self.on_some_event)
            ```
        """
        pass

    def restore_from_session(self, state: dict[str, Any]) -> Optional[list[Message]]:
        """Restore the Conversation Manager's state from a session.

        Args:
            state: Previous state of the conversation manager
        Returns:
            Optional list of messages to prepend to the agents messages. By default returns None.
        """
        if state.get("__name__") != self.__class__.__name__:
            raise ValueError("Invalid conversation manager state.")
        self.removed_message_count = state["removed_message_count"]
        return None

    def get_state(self) -> dict[str, Any]:
        """Get the current state of a Conversation Manager as a Json serializable dictionary."""
        return {
            "__name__": self.__class__.__name__,
            "removed_message_count": self.removed_message_count,
        }

    @abstractmethod
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Applies management strategy to the provided agent.

        Processes the conversation history to maintain appropriate size by modifying the messages list in-place.
        Implementations should handle message pruning, summarization, or other size management techniques to keep the
        conversation context within desired bounds.

        Args:
            agent: The agent whose conversation history will be manage.
                This list is modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass

    @abstractmethod
    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None, **kwargs: Any) -> None:
        """Called when the model's context window is exceeded.

        This method should implement the specific strategy for reducing the window size when a context overflow occurs.
        It is typically called after a ContextWindowOverflowException is caught.

        Implementations might use strategies such as:

        - Removing the N oldest messages
        - Summarizing older context
        - Applying importance-based filtering
        - Maintaining critical conversation markers

        Args:
            agent: The agent whose conversation history will be reduced.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass
