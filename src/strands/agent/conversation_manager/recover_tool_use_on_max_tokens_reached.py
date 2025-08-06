"""Shared utility for handling token limit recovery in conversation managers."""

import logging
from typing import TYPE_CHECKING

from ...hooks import MessageAddedEvent
from ...types.content import ContentBlock, Message
from ...types.exceptions import MaxTokensReachedException
from ...types.tools import ToolUse

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


async def recover_tool_use_on_max_tokens_reached(agent: "Agent", exception: MaxTokensReachedException) -> None:
    """Handle MaxTokensReachedException by cleaning up orphaned tool uses and adding corrected message.

    This function fixes incomplete tool uses that may occur when the model's response is truncated
    due to token limits. It:

    1. Inspects each content block in the incomplete message for invalid tool uses
    2. Replaces incomplete tool use blocks with informative text messages
    3. Preserves valid content blocks in the corrected message
    4. Adds the corrected message to the agent's conversation history

    Args:
        agent: The agent whose conversation will be updated with the corrected message.
        exception: The MaxTokensReachedException containing the incomplete message.
    """
    logger.info("handling MaxTokensReachedException - inspecting incomplete message for invalid tool uses")

    incomplete_message: Message = exception.incomplete_message
    logger.warning(f"incomplete message {incomplete_message}")

    if not incomplete_message["content"]:
        # Cannot correct invalid content block if content is empty
        raise exception

    valid_content: list[ContentBlock] = []
    has_corrected_content = False
    for content in incomplete_message["content"]:
        tool_use: ToolUse | None = content.get("toolUse")
        if not tool_use:
            valid_content.append(content)
            continue

        # Check if tool use is incomplete (missing or empty required fields)
        tool_name = tool_use.get("name")
        if not (tool_name and tool_use.get("input") and tool_use.get("toolUseId")):
            # Tool use is incomplete due to max_tokens truncation
            display_name = tool_name if tool_name else "<unknown>"
            logger.warning("tool_name=<%s> | replacing with error message due to max_tokens truncation.", display_name)

            valid_content.append(
                {
                    "text": f"The selected tool {display_name}'s tool use was incomplete due "
                    f"to maximum token limits being reached."
                }
            )
            has_corrected_content = True
        else:
            # ToolUse was invalid for an unknown reason. Cannot correct, return without modifying
            raise exception

    valid_message: Message = {"content": valid_content, "role": incomplete_message["role"]}
    agent.messages.append(valid_message)
    agent.hooks.invoke_callbacks(MessageAddedEvent(agent=agent, message=valid_message))
