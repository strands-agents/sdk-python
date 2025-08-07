"""Message recovery utilities for handling max token limit scenarios.

This module provides functionality to recover and clean up incomplete messages that occur
when model responses are truncated due to maximum token limits being reached. It specifically
handles cases where tool use blocks are incomplete or malformed due to truncation.
"""

import logging

from ..types.content import ContentBlock, Message
from ..types.tools import ToolUse

logger = logging.getLogger(__name__)


def recover_message_on_max_tokens_reached(message: Message) -> Message:
    """Recover and clean up incomplete messages when max token limits are reached.

    When a model response is truncated due to maximum token limits, tool use blocks may be
    incomplete or malformed. This function inspects the message content and:

    1. Identifies incomplete tool use blocks (missing name, input, or toolUseId)
    2. Replaces incomplete tool uses with informative error messages
    3. Preserves all valid content blocks (text and complete tool uses)
    4. Returns a cleaned message suitable for conversation history

    This recovery mechanism ensures that the conversation can continue gracefully even when
    model responses are truncated, providing clear feedback about what happened.

    Args:
        message: The potentially incomplete message from the model that was truncated
                due to max token limits.

    Returns:
        A cleaned Message with incomplete tool uses replaced by explanatory text content.
        The returned message maintains the same role as the input message.

    Example:
        If a message contains an incomplete tool use like:
        ```
        {"toolUse": {"name": "calculator"}}  # missing input and toolUseId
        ```

        It will be replaced with:
        ```
        {"text": "The selected tool calculator's tool use was incomplete due to maximum token limits being reached."}
        ```
    """
    logger.info("handling max_tokens stop reason - inspecting incomplete message for invalid tool uses")

    valid_content: list[ContentBlock] = []
    for content in message["content"] or []:
        tool_use: ToolUse | None = content.get("toolUse")
        if not tool_use:
            valid_content.append(content)
            continue

        # Check if tool use is incomplete (missing or empty required fields)
        tool_name = tool_use.get("name")
        if tool_name and tool_use.get("input") and tool_use.get("toolUseId"):
            # As far as we can tell, tool use is valid if this condition is true
            valid_content.append(content)
            continue

        # Tool use is incomplete due to max_tokens truncation
        display_name = tool_name if tool_name else "<unknown>"
        logger.warning("tool_name=<%s> | replacing with error message due to max_tokens truncation.", display_name)

        valid_content.append(
            {
                "text": f"The selected tool {display_name}'s tool use was incomplete due "
                f"to maximum token limits being reached."
            }
        )

    return {"content": valid_content, "role": message["role"]}
