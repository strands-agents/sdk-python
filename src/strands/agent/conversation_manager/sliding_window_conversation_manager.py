"""Sliding window conversation history management."""

import json
import logging
from typing import List, Optional, cast

from ...types.content import ContentBlock, Message, Messages
from ...types.exceptions import ContextWindowOverflowException
from ...types.tools import ToolResult
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
            if not any("toolResult" in content for content in messages[-1]["content"]):
                messages.pop()

        # remove any dangling assistant messages with ToolUse
        if len(messages) > 0 and is_assistant_message(messages[-1]):
            if any("toolUse" in content for content in messages[-1]["content"]):
                messages.pop()
                # remove remaining dangling user messages with no ToolResult after we popped off an assistant message
                if len(messages) > 0 and is_user_message(messages[-1]):
                    if not any("toolResult" in content for content in messages[-1]["content"]):
                        messages.pop()

    def reduce_context(self, messages: Messages, e: Optional[Exception] = None) -> None:
        """Trim the oldest messages to reduce the conversation context size.

        The method handles special cases where tool results need to be converted to regular content blocks to maintain
        conversation coherence after trimming.

        Args:
            messages: The messages to reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
                Such as when the conversation is already minimal or when tool result messages cannot be properly
                converted.
        """
        # If the number of messages is less than the window_size, then we default to 2, otherwise, trim to window size
        trim_index = 2 if len(messages) <= self.window_size else len(messages) - self.window_size

        # Throw if we cannot trim any messages from the conversation
        if trim_index >= len(messages):
            raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # If the message at the cut index has ToolResultContent, then we map that to ContentBlock. This gets around the
        # limitation of needing ToolUse and ToolResults to be paired.
        if any("toolResult" in content for content in messages[trim_index]["content"]):
            if len(messages[trim_index]["content"]) == 1:
                messages[trim_index]["content"] = self._map_tool_result_content(
                    cast(ToolResult, messages[trim_index]["content"][0]["toolResult"])
                )

            # If there is more content than just one ToolResultContent, then we cannot cut at this index.
            else:
                raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # Overwrite message history
        messages[:] = messages[trim_index:]

    def _map_tool_result_content(self, tool_result: ToolResult) -> List[ContentBlock]:
        """Convert a ToolResult to a list of standard ContentBlocks.

        This method transforms tool result content into standard content blocks that can be preserved when trimming the
        conversation history.

        Args:
            tool_result: The ToolResult to convert.

        Returns:
            A list of content blocks representing the tool result.
        """
        contents = []
        text_content = "Tool Result Status: " + tool_result["status"] if tool_result["status"] else ""

        for tool_result_content in tool_result["content"]:
            if "text" in tool_result_content:
                text_content = "\nTool Result Text Content: " + tool_result_content["text"] + f"\n{text_content}"
            elif "json" in tool_result_content:
                text_content = (
                    "\nTool Result JSON Content: " + json.dumps(tool_result_content["json"]) + f"\n{text_content}"
                )
            elif "image" in tool_result_content:
                contents.append(ContentBlock(image=tool_result_content["image"]))
            elif "document" in tool_result_content:
                contents.append(ContentBlock(document=tool_result_content["document"]))
            else:
                logger.warning("unsupported content type")
        contents.append(ContentBlock(text=text_content))
        return contents
