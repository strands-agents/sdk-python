"""Sliding window conversation history management."""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...agent.agent import Agent

from ...types.content import Messages
from ...types.exceptions import ContextWindowOverflowException
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class SlidingWindowConversationManager(ConversationManager):
    """Implements a sliding window strategy for managing conversation history.

    This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids
    invalid window states.
    """

    def __init__(self, window_size: int = 40, should_truncate_results: bool = True):
        """Initialize the sliding window conversation manager.

        Args:
            window_size: Maximum number of messages to keep in the agent's history.
                Defaults to 40 messages.
            should_truncate_results: Truncate tool results when a message is too large for the model's context window
        """
        super().__init__()
        self.window_size = window_size
        self.should_truncate_results = should_truncate_results

    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply the sliding window to the agent's messages array to maintain a manageable history size.

        This method is called after every event loop cycle to apply a sliding window if the message count
        exceeds the window size.

        Args:
            agent: The agent whose messages will be managed.
                This list is modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        messages = agent.messages

        if len(messages) <= self.window_size:
            logger.debug(
                "message_count=<%s>, window_size=<%s> | skipping context reduction", len(messages), self.window_size
            )
            return
        self.reduce_context(agent)

    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None, **kwargs: Any) -> None:
        """Trim the oldest messages to reduce the conversation context size.

        The method handles special cases where trimming the messages leads to:
         - toolResult with no corresponding toolUse
         - toolUse with no corresponding toolResult

        Args:
            agent: The agent whose messages will be reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
                Such as when the conversation is already minimal or when tool result messages cannot be properly
                converted.
        """
        messages = agent.messages

        # Try to truncate the tool result first
        last_message_idx_with_tool_results = self._find_last_message_with_tool_results(messages)
        if last_message_idx_with_tool_results is not None and self.should_truncate_results:
            logger.debug(
                "message_index=<%s> | found message with tool results at index", last_message_idx_with_tool_results
            )
            results_truncated = self._truncate_tool_results(messages, last_message_idx_with_tool_results)
            if results_truncated:
                logger.debug("message_index=<%s> | tool results truncated", last_message_idx_with_tool_results)
                return

        # Try to trim index id when tool result cannot be truncated anymore
        # If the number of messages is less than the window_size, then we default to 2, otherwise, trim to window size
        trim_index = 2 if len(messages) <= self.window_size else len(messages) - self.window_size

        # Find the next valid trim_index
        while trim_index < len(messages):
            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any("toolResult" in content for content in messages[trim_index]["content"])
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    any("toolUse" in content for content in messages[trim_index]["content"])
                    and trim_index + 1 < len(messages)
                    and not any("toolResult" in content for content in messages[trim_index + 1]["content"])
                )
            ):
                trim_index += 1
            else:
                break
        else:
            # If we didn't find a valid trim_index, then we throw
            raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # trim_index represents the number of messages being removed from the agents messages array
        self.removed_message_count += trim_index

        # Overwrite message history
        messages[:] = messages[trim_index:]

    def _truncate_tool_results(self, messages: Messages, msg_idx: int) -> bool:
        """Truncate tool results in a message to reduce context size.

        When a message contains tool results that are too large for the model's context window, this function
        replaces the content of those tool results with a simple error message.

        Args:
            messages: The conversation message history.
            msg_idx: Index of the message containing tool results to truncate.

        Returns:
            True if any changes were made to the message, False otherwise.
        """
        if msg_idx >= len(messages) or msg_idx < 0:
            return False

        message = messages[msg_idx]
        changes_made = False
        tool_result_too_large_message = "The tool result was too large!"
        for i, content in enumerate(message.get("content", [])):
            if isinstance(content, dict) and "toolResult" in content:
                tool_result_content_text = next(
                    (item["text"] for item in content["toolResult"]["content"] if "text" in item),
                    "",
                )
                # make the overwriting logic togglable
                if (
                    message["content"][i]["toolResult"]["status"] == "error"
                    and tool_result_content_text == tool_result_too_large_message
                ):
                    logger.info("ToolResult has already been updated, skipping overwrite")
                    return False
                # Update status to error with informative message
                message["content"][i]["toolResult"]["status"] = "error"
                message["content"][i]["toolResult"]["content"] = [{"text": tool_result_too_large_message}]
                changes_made = True

        return changes_made

    def _find_last_message_with_tool_results(self, messages: Messages) -> Optional[int]:
        """Find the index of the last message containing tool results.

        This is useful for identifying messages that might need to be truncated to reduce context size.

        Args:
            messages: The conversation message history.

        Returns:
            Index of the last message with tool results, or None if no such message exists.
        """
        # Iterate backwards through all messages (from newest to oldest)
        for idx in range(len(messages) - 1, -1, -1):
            # Check if this message has any content with toolResult
            current_message = messages[idx]
            has_tool_result = False

            for content in current_message.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    has_tool_result = True
                    break

            if has_tool_result:
                return idx

        return None
