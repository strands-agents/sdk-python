"""Abstract interface for conversation history management."""

import copy
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict, Union

from ..._middleware.stages import InvokeModelContext
from ..._middleware.types import MiddlewareInputHandler
from ...hooks.registry import HookProvider, HookRegistry
from ...models._defaults import DEFAULT_COMPRESSION_THRESHOLD
from ...types.content import Message, split_system_prompt

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


class ProactiveCompressionConfig(TypedDict, total=False):
    """Configuration for proactive compression when passed as an object.

    Attributes:
        compression_threshold: Ratio of context window usage that triggers proactive compression.
            Value between 0 (exclusive) and 1 (inclusive).
            Defaults to 0.7 (compress when 70% of the context window is used).
    """

    compression_threshold: float


class ConversationManager(ABC, HookProvider):
    """Abstract base class for managing conversation history.

    This class provides an interface for implementing conversation management strategies to control the size of message
    arrays/conversation histories, helping to:

    - Manage memory usage
    - Control context length
    - Maintain relevant conversation state

    ConversationManager implements the HookProvider protocol so derived classes can register hooks for agent
    lifecycle events when needed.

    The primary responsibility of a ConversationManager is overflow recovery: when the model encounters a context
    window overflow, :meth:`reduce_context` is called with ``e`` set and MUST reduce the history enough for the next
    model call to succeed.

    Subclasses can enable proactive compression by passing ``proactive_compression`` in the constructor.
    When enabled, invoke-model middleware checks the effective model's projected input tokens against its
    context window and calls :meth:`reduce_context` (without ``e``) when the threshold is exceeded. This
    is a best-effort operation: errors are swallowed so the model call can still proceed.

    Example:
        ```python
        # Enable proactive compression with default threshold (0.7)
        SlidingWindowConversationManager(window_size=50, proactive_compression=True)

        # Enable proactive compression with custom threshold
        SummarizingConversationManager(proactive_compression={"compression_threshold": 0.8})
        ```
    """

    def __init__(self, *, proactive_compression: Union[bool, "ProactiveCompressionConfig", None] = None) -> None:
        """Initialize the ConversationManager.

        Args:
            proactive_compression: Enable proactive context compression before the model call.
                - ``True``: compress when 70% of the context window is used (default threshold).
                - ``{"compression_threshold": float}``: compress at the specified ratio (0, 1].
                - ``False`` or ``None``: disabled, only reactive overflow recovery is used.

        Raises:
            ValueError: If compression_threshold is not in the valid range (0, 1].

        Attributes:
          removed_message_count: The messages that have been removed from the agents messages array.
              These represent messages provided by the user or LLM that have been removed, not messages
              included by the conversation manager through something like summarization.
        """
        # Resolve the threshold from proactive_compression parameter
        if proactive_compression is True:
            threshold: float | None = DEFAULT_COMPRESSION_THRESHOLD
        elif isinstance(proactive_compression, dict):
            threshold = proactive_compression.get("compression_threshold", DEFAULT_COMPRESSION_THRESHOLD)
        else:
            threshold = None

        if threshold is not None and (threshold <= 0 or threshold > 1):
            raise ValueError(f"compression_threshold must be between 0 (exclusive) and 1 (inclusive), got {threshold}")

        self.removed_message_count = 0
        self._compression_threshold = threshold

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register conversation-manager-specific hooks.

        The base manager has no hooks. Derived classes may override this method to subscribe to
        lifecycle events and can call the base implementation without additional requirements.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """

    def _proactive_compression_middleware(self) -> MiddlewareInputHandler:
        """Build middleware that compresses against the effective model's context window."""

        async def middleware(context: InvokeModelContext) -> InvokeModelContext:
            if self._compression_threshold is None:
                return context

            try:
                input_tokens = await self._count_input_tokens(context)
                context.projected_input_tokens = input_tokens
                ratio = context.model.estimate_utilization(input_tokens)
            except Exception:
                logger.debug("proactive compression measurement failed, proceeding with model call", exc_info=True)
                return context

            if ratio < self._compression_threshold:
                return context

            logger.debug(
                "projected_tokens=<%s>, context_window_limit=<%s>, ratio=<%.2f>, compression_threshold=<%s>"
                " | compression threshold exceeded, reducing context",
                input_tokens,
                context.model.context_window_limit,
                ratio,
                self._compression_threshold,
            )
            try:
                self.reduce_context(agent=context.agent)
            except Exception:
                logger.debug("proactive compression failed, proceeding with model call", exc_info=True)

            try:
                context.messages = copy.deepcopy(context.agent.messages)
                context.projected_input_tokens = None
                context.projected_input_tokens = await self._count_input_tokens(context)
            except Exception:
                logger.debug("proactive compression refresh failed, proceeding with model call", exc_info=True)
            return context

        return middleware

    @staticmethod
    async def _count_input_tokens(context: InvokeModelContext) -> int:
        """Count the middleware request with its effective model."""
        system_prompt, system_prompt_content = split_system_prompt(context.system_prompt)
        return await context.model.count_tokens(
            context.messages,
            tool_specs=list(context.tool_specs),
            system_prompt=system_prompt,
            system_prompt_content=system_prompt_content,
        )

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
        """Reduce the conversation history.

        Called in two scenarios:
        1. **Reactive** (e is set): A context window overflow occurred. The implementation
           MUST remove enough history for the next model call to succeed, or re-raise the error.
        2. **Proactive** (e is None): The compression threshold was exceeded. This is best-effort —
           returning without reduction or raising is acceptable; the model call proceeds regardless.

        Implementations should modify ``agent.messages`` in-place.

        Args:
            agent: The agent whose conversation history will be reduced.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.
                When set, this is a reactive overflow recovery call — the implementation MUST
                reduce enough history for the next model call to succeed.
                When None, this is a proactive compression call — best-effort reduction to avoid
                hitting the context window limit.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        pass
