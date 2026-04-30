"""Abstract interface for conversation history management."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ...hooks.events import BeforeModelCallEvent
from ...hooks.registry import HookProvider, HookRegistry
from ...types.content import Message

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW_LIMIT = 200_000


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

    Optionally, a manager can enable proactive compression by setting ``compression_threshold``
    in the constructor. When set, the base class registers a ``BeforeModelCallEvent`` hook that
    checks projected input tokens against the model's context window limit and calls
    :meth:`reduce_on_threshold` when the threshold is exceeded.

    Example:
        ```python
        class MyConversationManager(ConversationManager):
            def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
                super().register_hooks(registry, **kwargs)
                # Register additional hooks here
        ```
    """

    def __init__(self, *, compression_threshold: float | None = None) -> None:
        """Initialize the ConversationManager.

        Args:
            compression_threshold: Ratio of context window usage that triggers proactive compression.
                Value between 0 (exclusive) and 1 (inclusive). For example, 0.7 means compress when 70%
                of the context window is used. When not set, proactive compression is disabled and only
                reactive overflow recovery is used.

        Raises:
            ValueError: If compression_threshold is not in the valid range (0, 1].

        Attributes:
          removed_message_count: The messages that have been removed from the agents messages array.
              These represent messages provided by the user or LLM that have been removed, not messages
              included by the conversation manager through something like summarization.
        """
        if compression_threshold is not None and (compression_threshold <= 0 or compression_threshold > 1):
            raise ValueError(
                f"compression_threshold must be between 0 (exclusive) and 1 (inclusive), got {compression_threshold}"
            )

        self.removed_message_count = 0
        self._compression_threshold = compression_threshold
        self._context_window_limit_warned = False

    def reduce_on_threshold(self, agent: "Agent", **kwargs: Any) -> bool:
        """Proactively reduce the conversation history before a model call.

        Called when projected input tokens exceed the configured compression_threshold
        of the model's context window limit. Subclasses implement this to reduce
        context before the model call, avoiding overflow errors.

        The base class catches any exceptions raised by this method and logs them
        at debug level, so subclass implementations do not need to defensively
        swallow errors — they can let them propagate. When an exception occurs,
        the return value is never observed by the caller.

        The default implementation returns False. Subclasses that support proactive
        compression should override this method.

        Args:
            agent: The agent whose conversation history will be reduced.
                The agent's messages list should be modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            True if the history was reduced, False otherwise. Only observed on success;
            if the method raises, the base class catches the exception and the return
            value is ignored.
        """
        return False

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for agent lifecycle events.

        When ``compression_threshold`` is configured and the subclass overrides
        ``reduce_on_threshold``, registers a ``BeforeModelCallEvent`` hook for
        proactive compression.

        Derived classes that override this method must call the base implementation to ensure proper hook
        registration chain.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        if self._compression_threshold is None:
            return

        # Check if the subclass actually overrides reduce_on_threshold
        has_override = type(self).reduce_on_threshold is not ConversationManager.reduce_on_threshold
        if not has_override:
            logger.warning(
                "conversation_manager=<%s> | compression_threshold is configured but reduce_on_threshold is not"
                " implemented, proactive compression is disabled",
                type(self).__name__,
            )
            return

        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call_threshold)

    def _on_before_model_call_threshold(self, event: BeforeModelCallEvent) -> None:
        """Handle BeforeModelCallEvent for proactive compression.

        Args:
            event: The before model call event.
        """
        context_window_limit = event.agent.model.context_window_limit
        if context_window_limit is None:
            context_window_limit = DEFAULT_CONTEXT_WINDOW_LIMIT
            if not self._context_window_limit_warned:
                self._context_window_limit_warned = True
                logger.warning(
                    "context_window_limit=<None>, default=<%s>"
                    " | context_window_limit is not set on the model, using default"
                    " | set context_window_limit in your model config for accurate threshold checks",
                    DEFAULT_CONTEXT_WINDOW_LIMIT,
                )

        if event.projected_input_tokens is None:
            logger.debug("projected_input_tokens=<None> | skipping proactive compression")
            return

        ratio = event.projected_input_tokens / context_window_limit
        if ratio >= self._compression_threshold:  # type: ignore[operator]
            logger.debug(
                "projected_tokens=<%s>, limit=<%s>, ratio=<%.2f>, compression_threshold=<%s>"
                " | compression threshold exceeded, reducing context",
                event.projected_input_tokens,
                context_window_limit,
                ratio,
                self._compression_threshold,
            )
            try:
                self.reduce_on_threshold(agent=event.agent)
            except Exception:
                logger.debug("proactive compression failed, will proceed with model call", exc_info=True)

    def restore_from_session(self, state: dict[str, Any]) -> list[Message] | None:
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
    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None:
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
