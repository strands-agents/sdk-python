"""Conversation manager that selectively prunes messages using configurable strategies."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from typing_extensions import TypedDict, override

from ...types.content import Message, Messages
from ...types.exceptions import ContextWindowOverflowException
from .conversation_manager import ConversationManager

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


class MessageContext(TypedDict):
    """Context information for a specific message in a conversation.

    This type represents the context data returned by get_message_context(),
    providing metadata about a message that can be used for pruning decisions.

    Attributes:
        token_count: Estimated number of tokens in the message.
        has_tool_use: Whether the message contains tool use content.
        has_tool_result: Whether the message contains tool result content.
        message_index: The index position of the message in the conversation.
        total_messages: Total number of messages in the conversation.
    """

    token_count: int
    has_tool_use: bool
    has_tool_result: bool
    message_index: int
    total_messages: int


class PruningStrategy(ABC):
    """Abstract interface for message pruning strategies.

    This class defines the contract that all pruning strategies must implement.
    Strategies can selectively compress or remove messages based on their own
    criteria while preserving conversation integrity.
    """

    @abstractmethod
    def should_prune_message(self, message: Message, context: MessageContext) -> bool:
        """Determine if a message should be pruned.

        Args:
            message: The message to evaluate for pruning
            context: Context information including message age, token count, etc.

        Returns:
            True if the message should be pruned, False otherwise
        """
        pass

    @abstractmethod
    def prune_message(self, message: Message, agent: "Agent") -> Optional[Message]:
        """Prune a message, returning the compressed version or None to remove.

        Args:
            message: The message to prune
            agent: The agent instance for context and LLM access

        Returns:
            Compressed message, or None to remove the message entirely
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this pruning strategy.

        Returns:
            The strategy name
        """
        pass


class PruningContext:
    """Context information for pruning decisions.

    This class provides rich context information to pruning strategies,
    enabling intelligent decision-making about which messages to prune
    and how to prune them.
    """

    def __init__(self, messages: Messages, agent: "Agent"):
        """Initialize pruning context.

        Args:
            messages: The conversation messages to analyze
            agent: The agent instance for additional context
        """
        self.messages = messages
        self.agent = agent
        self.token_counts = self._estimate_token_counts()

    def get_message_context(self, index: int) -> MessageContext:
        """Get context information for a specific message.

        Args:
            index: The index of the message in the conversation

        Returns:
            MessageContext containing context information for the message

        Raises:
            IndexError: If the message index is out of range
        """
        if index < 0 or index >= len(self.messages):
            raise IndexError(f"Message index {index} out of range")

        return {
            "token_count": self.token_counts[index],
            "has_tool_use": self._has_tool_use(self.messages[index]),
            "has_tool_result": self._has_tool_result(self.messages[index]),
            "message_index": index,
            "total_messages": len(self.messages),
        }

    def _estimate_token_counts(self) -> List[int]:
        """Estimate token count for each message.

        This provides a rough estimation of token usage for each message.
        The estimation can be enhanced with actual tokenization in the future.

        Returns:
            List of estimated token counts for each message
        """
        token_counts = []
        for message in self.messages:
            count = 0
            for content in message.get("content", []):
                if "text" in content:
                    # Rough token estimation: ~1.3 tokens per word
                    count += int(len(content["text"].split()) * 1.3)
                elif "toolResult" in content:
                    result_content = content["toolResult"].get("content", [])
                    for result_item in result_content:
                        if "text" in result_item:
                            count += int(len(result_item["text"].split()) * 1.3)
                        elif "json" in result_item:
                            count += int(len(str(result_item["json"]).split()) * 1.3)
                elif "toolUse" in content:
                    # Rough estimate for tool use overhead
                    count += 50
                    # Add input parameter tokens
                    input_str = str(content["toolUse"].get("input", {}))
                    count += int(len(input_str.split()) * 1.3)
            token_counts.append(int(count))
        return token_counts

    def _has_tool_use(self, message: Message) -> bool:
        """Check if message contains tool use.

        Args:
            message: The message to check

        Returns:
            True if the message contains tool use, False otherwise
        """
        return any("toolUse" in content for content in message.get("content", []))

    def _has_tool_result(self, message: Message) -> bool:
        """Check if message contains tool result.

        Args:
            message: The message to check

        Returns:
            True if the message contains tool result, False otherwise
        """
        return any("toolResult" in content for content in message.get("content", []))


class PruningConversationManager(ConversationManager):
    """Conversation manager that selectively prunes messages using configurable strategies.

    Unlike summarization which collapses multiple messages into one, pruning returns
    a list of messages where some have been compressed, removed, or truncated while
    others remain intact. This preserves conversation structure and flow.
    """

    def __init__(
        self,
        pruning_strategies: List[PruningStrategy],
        preserve_initial_messages: int = 1,
        preserve_recent_messages: int = 2,
        enable_proactive_pruning: bool = True,
        pruning_threshold: float = 0.7,
        context_window_size: int = 200000,
    ):
        """Initialize the pruning conversation manager.

        Args:
            pruning_strategies: List of strategies to apply for message pruning
            preserve_initial_messages: Number of initial messages to never prune
            preserve_recent_messages: Number of recent messages to never prune
            enable_proactive_pruning: Whether to prune proactively based on threshold
            pruning_threshold: Context usage threshold to trigger proactive pruning (0.1-1.0)
            context_window_size: Maximum context window size in tokens (default: 200000)
        """
        super().__init__()
        self.pruning_strategies = pruning_strategies
        self.preserve_recent_messages = preserve_recent_messages
        self.preserve_initial_messages = preserve_initial_messages
        self.enable_proactive_pruning = enable_proactive_pruning
        self.pruning_threshold = max(0.1, min(1.0, pruning_threshold))
        self.context_window_size = context_window_size

    @override
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply pruning management strategy.

        For proactive pruning, this method checks if the conversation has grown
        beyond the configured threshold and applies pruning strategies if needed.

        Args:
            agent: The agent whose conversation history will be managed
            **kwargs: Additional keyword arguments for future extensibility
        """
        if self.enable_proactive_pruning and self._should_prune_proactively(agent):
            logger.debug("Applying proactive pruning based on threshold")
            self.reduce_context(agent, **kwargs)

    @override
    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None, **kwargs: Any) -> None:
        """Reduce context through selective message pruning.

        This method applies the configured pruning strategies to reduce the conversation
        context size while preserving important messages and conversation structure.

        Args:
            agent: The agent whose conversation history will be reduced
            e: The exception that triggered the context reduction, if any
            **kwargs: Additional keyword arguments for future extensibility

        Raises:
            ContextWindowOverflowException: If the context cannot be pruned sufficiently
        """
        if len(agent.messages) == 0:
            raise ContextWindowOverflowException("No messages to prune")

        original_messages = agent.messages.copy()
        original_count = len(original_messages)

        try:
            pruned_messages, removed_count = self._prune_messages(agent.messages, agent)

            # Validate that pruning actually reduced context
            if self._validate_pruning_effectiveness(original_messages, pruned_messages, agent):
                agent.messages[:] = pruned_messages
                self.removed_message_count += removed_count

                logger.info(
                    "Pruning completed: %d -> %d messages (%d removed)",
                    original_count,
                    len(pruned_messages),
                    removed_count,
                )
            else:
                # Fallback to more aggressive pruning or raise exception
                self._handle_pruning_failure(agent, e)

        except Exception as pruning_error:
            logger.error("Pruning failed: %s", pruning_error)
            raise pruning_error from e

    def _prune_messages(self, messages: Messages, agent: "Agent") -> tuple[Messages, int]:
        """Apply pruning strategies to messages.

        When the token threshold is exceeded, all messages outside the preserved ranges
        (initial and recent) will be pruned according to the configured strategies.

        Args:
            messages: The messages to prune
            agent: The agent instance for context

        Returns:
            A tuple of (pruned messages list, count of removed messages)

        Raises:
            ContextWindowOverflowException: If there are insufficient messages to prune safely
        """
        total_preserved = self.preserve_recent_messages + self.preserve_initial_messages
        if len(messages) <= total_preserved:
            logger.debug("Too few messages to prune safely")
            raise ContextWindowOverflowException("Insufficient messages for pruning")

        # Create pruning context
        context = PruningContext(messages, agent)

        # Determine which messages can be pruned
        # Exclude initial messages (0 to preserve_initial_messages-1)
        # Exclude recent messages (len(messages) - preserve_recent_messages to end)
        min_prunable_index = self.preserve_initial_messages
        max_prunable_index = len(messages) - self.preserve_recent_messages

        pruned_messages = []
        removed_count = 0

        for i, message in enumerate(messages):
            # Always preserve initial messages
            if i < min_prunable_index:
                pruned_messages.append(message)
                continue

            # Always preserve recent messages
            if i >= max_prunable_index:
                pruned_messages.append(message)
                continue

            # For messages in the prunable range
            message_context = context.get_message_context(i)
            should_prune = False

            # Apply pruning strategies to determine if message should be pruned
            for strategy in self.pruning_strategies:
                if strategy.should_prune_message(message, message_context):
                    should_prune = True
                    break

            if should_prune:
                # Try to prune the message
                pruned_message = None
                for strategy in self.pruning_strategies:
                    if strategy.should_prune_message(message, message_context):
                        pruned_message = strategy.prune_message(message, agent)
                        break

                if pruned_message is not None:
                    # Message was compressed
                    pruned_messages.append(pruned_message)
                else:
                    # Message was removed entirely - don't add to pruned_messages
                    removed_count += 1
                    # Note: we don't append anything to pruned_messages, effectively removing it
            else:
                # Keep message unchanged
                pruned_messages.append(message)

        return pruned_messages, removed_count

    def _should_prune_proactively(self, agent: "Agent") -> bool:
        """Determine if proactive pruning should be triggered.

        Args:
            agent: The agent to evaluate

        Returns:
            True if proactive pruning should be applied
        """
        if not agent.messages:
            return False

        # Create pruning context to calculate total tokens
        context = PruningContext(agent.messages, agent)
        total_tokens = sum(context.token_counts)

        # Apply threshold to context window size
        threshold_tokens = self.context_window_size * self.pruning_threshold

        logger.debug(
            "Proactive pruning check: %d tokens / %d threshold (%d limit * %s threshold)",
            total_tokens,
            threshold_tokens,
            self.context_window_size,
            self.pruning_threshold,
        )

        return total_tokens > threshold_tokens

    def _validate_pruning_effectiveness(
        self, original_messages: Messages, pruned_messages: Messages, agent: "Agent"
    ) -> bool:
        """Validate that pruning actually reduced context size.

        Args:
            original_messages: The original message list
            pruned_messages: The pruned message list
            agent: The agent instance

        Returns:
            True if pruning was effective, False otherwise
        """
        # Calculate total estimated tokens for original and pruned messages
        original_context = PruningContext(original_messages, agent)
        pruned_context = PruningContext(pruned_messages, agent)

        original_tokens = sum(original_context.token_counts)
        pruned_tokens = sum(pruned_context.token_counts)

        logger.debug("Token comparison: original=%d, pruned=%d", original_tokens, pruned_tokens)

        return pruned_tokens < original_tokens

    def _handle_pruning_failure(self, agent: "Agent", e: Optional[Exception]) -> None:
        """Handle cases where pruning fails to reduce context sufficiently.

        Args:
            agent: The agent instance
            e: The original exception that triggered pruning, if any

        Raises:
            ContextWindowOverflowException: When pruning fails to reduce context
        """
        logger.error("Pruning failed to reduce context sufficiently")

        # Could implement more aggressive fallback strategies here
        # For now, raise the original exception
        if e:
            raise e
        else:
            raise ContextWindowOverflowException("Pruning failed to reduce context")

    @override
    def get_state(self) -> Dict[str, Any]:
        """Get the current state including pruning statistics.

        Returns:
            Dictionary containing the manager's state for session persistence
        """
        return super().get_state()

    @override
    def restore_from_session(self, state: Dict[str, Any]) -> Optional[List[Message]]:
        """Restore pruning manager state from session.

        Args:
            state: The state dictionary to restore from

        Returns:
            None (no messages to prepend)
        """
        super().restore_from_session(state)
        return None
