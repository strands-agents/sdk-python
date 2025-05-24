"""Sliding window conversation history management."""

import logging
from typing import List, Optional

from ...types.content import Message, Messages
from ...types.exceptions import ContextWindowOverflowException
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


def is_user_message(message: Message) -> bool:
    """Check if a message is from a user.

    Args:
        message: The message object to check.

    Returns:
        True if the message has the user role, False otherwise.
    """
    return message["role"] == "user"


def is_assistant_message(message: Message) -> bool:
    """Check if a message is from an assistant.

    Args:
        message: The message object to check.

    Returns:
        True if the message has the assistant role, False otherwise.
    """
    return message["role"] == "assistant"


def has_tool_use(message: Message) -> bool:
    """Check if a message contains toolUse content."""
    return any("toolUse" in content for content in message["content"])


def has_tool_result(message: Message) -> bool:
    """Check if a message contains toolResult content."""
    return any("toolResult" in content for content in message["content"])


def get_tool_use_ids(message: Message) -> List[str]:
    """Get all toolUse IDs from a message."""
    ids = []
    for content in message["content"]:
        if "toolUse" in content:
            ids.append(content["toolUse"]["toolUseId"])
    return ids


def get_tool_result_ids(message: Message) -> List[str]:
    """Get all toolResult IDs from a message."""
    ids = []
    for content in message["content"]:
        if "toolResult" in content:
            ids.append(content["toolResult"]["toolUseId"])
    return ids


class SlidingWindowConversationManager(ConversationManager):
    """Implements a sliding window strategy for managing conversation history.

    This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids
    invalid window states.
    """

    def __init__(self, window_size: int = 40):
        """Initialize the sliding window conversation manager.

        Args:
            window_size: Maximum number of messages to keep in history.
                Defaults to 40 messages.
        """
        self.window_size = window_size

    def apply_management(self, messages: Messages) -> None:
        """Apply the sliding window to the messages array to maintain a manageable history size.

        This method is called after every event loop cycle, as the messages array may have been modified with tool
        results and assistant responses. It first removes any dangling messages that might create an invalid
        conversation state, then applies the sliding window if the message count exceeds the window size.

        Special handling is implemented to ensure we don't leave a user message with toolResult
        as the first message in the array. It also ensures that all toolUse blocks have corresponding toolResult
        blocks to maintain conversation coherence.

        Args:
            messages: The messages to manage.
                This list is modified in-place.
        """
        self._remove_dangling_messages(messages)

        if len(messages) <= self.window_size:
            logger.debug(
                "window_size=<%s>, message_count=<%s> | skipping context reduction", len(messages), self.window_size
            )
            return
        self.reduce_context(messages)

    def _remove_dangling_messages(self, messages: Messages) -> None:
        """Remove dangling messages that would create an invalid conversation state.

        After the event loop cycle is executed, we expect the messages array to end with either an assistant tool use
        request followed by the pairing user tool result or an assistant response with no tool use request. If the
        event loop cycle fails, we may end up in an invalid message state, and so this method will remove problematic
        messages from the end of the array.

        This method handles two specific cases:

        - User with no tool result: Indicates that event loop failed to generate an assistant tool use request
        - Assistant with tool use request: Indicates that event loop failed to generate a pairing user tool result

        Args:
            messages: The messages to clean up.
                This list is modified in-place.
        """
        # remove any dangling user messages with no ToolResult
        if len(messages) > 0 and is_user_message(messages[-1]):
            if not has_tool_result(messages[-1]):
                messages.pop()

        # remove any dangling assistant messages with ToolUse
        if len(messages) > 0 and is_assistant_message(messages[-1]):
            if has_tool_use(messages[-1]):
                messages.pop()
                # remove remaining dangling user messages with no ToolResult after we popped off an assistant message
                if len(messages) > 0 and is_user_message(messages[-1]):
                    if not has_tool_result(messages[-1]):
                        messages.pop()

    def reduce_context(self, messages: Messages, e: Optional[Exception] = None) -> None:
        """Trim the oldest messages to reduce the conversation context size.

        The method ensures that tool use/result pairs are preserved together. If a cut would separate
        a toolUse from its corresponding toolResult, it adjusts the cut point to include both.

        Args:
            messages: The messages to reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
        """
        # Calculate basic trim index
        trim_index = 2 if len(messages) <= self.window_size else len(messages) - self.window_size

        # Throw if we cannot trim any messages from the conversation
        if trim_index >= len(messages):
            raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # Find a safe cutting point that preserves tool use/result pairs
        safe_trim_index = self._find_safe_trim_index(messages, trim_index)

        # If we couldn't find a safe trim point within bounds, fall back to basic trim
        if safe_trim_index >= len(messages):
            logger.warning(
                "safe_trim_index=<%d>, messages_length=<%d> | could not find safe trim point | "
                "falling back to basic trim index",
                safe_trim_index,
                len(messages),
            )
            safe_trim_index = trim_index

        # Overwrite message history
        messages[:] = messages[safe_trim_index:]

    def _find_safe_trim_index(self, messages: Messages, initial_trim_index: int) -> int:
        """Find a safe cutting point that preserves tool use/result pairs.

        This method ensures that tool use/result pairs are not separated by the trim.
        It adjusts the trim index to keep related tool interactions together.

        Args:
            messages: The complete message history
            initial_trim_index: The initial trim index based on window size

        Returns:
            A safe trim index that preserves tool use/result pairs
        """
        # Build a map of tool IDs to their message indices
        tool_use_indices = {}  # toolUseId -> message index
        tool_result_indices = {}  # toolUseId -> message index

        for i, message in enumerate(messages):
            for tool_id in get_tool_use_ids(message):
                tool_use_indices[tool_id] = i
            for tool_id in get_tool_result_ids(message):
                tool_result_indices[tool_id] = i

        # Start from the initial trim index
        safe_index = initial_trim_index

        # Adjust if we would cut in the middle of a tool use/result pair
        for tool_id, use_idx in tool_use_indices.items():
            if tool_id in tool_result_indices:
                result_idx = tool_result_indices[tool_id]
                # If the pair would be split by the cut
                if use_idx < safe_index <= result_idx:
                    # Move the cut to before the tool use to keep the pair together
                    safe_index = min(safe_index, use_idx)
                elif result_idx < safe_index < use_idx:
                    # This shouldn't happen in valid conversations
                    logger.warning("tool_id=<%s> | found toolResult before toolUse", tool_id)

        return safe_index
