"""Hook provider for correcting incomplete tool uses due to token limits.

This module provides the CorrectToolUseHookProvider class, which handles scenarios where
the model's response is truncated due to maximum token limits, resulting in incomplete
or malformed tool use entries. The provider automatically corrects these issues to allow
the agent conversation to continue gracefully.
"""

import logging
from typing import Any

from strands.experimental.hooks.events import EventLoopFailureEvent
from strands.hooks import HookProvider, HookRegistry, MessageAddedEvent
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import MaxTokensReachedException
from strands.types.tools import ToolUse

logger = logging.getLogger(__name__)


class CorrectToolUseHookProvider(HookProvider):
    """Hook provider that handles MaxTokensReachedException by fixing incomplete tool uses.

    This hook provider is triggered when a MaxTokensReachedException occurs during event loop execution.
    When the model's response is truncated due to token limits, tool use entries may be incomplete
    or missing required fields (name, input, toolUseId).

    The provider fixes these issues by:

    1. Inspecting each content block in the incomplete message for invalid tool uses
    2. Replacing incomplete tool use blocks with informative text messages
    3. Preserving valid content blocks in the corrected message
    4. Adding the corrected message to the agent's conversation history
    5. Allowing the event loop to continue processing

    If a tool use is invalid for unknown reasons, not due to empty fields, the hook
    allows the original exception to propagate to avoid unsafe recovery attempts.
    """

    def register_hooks(self, registry: "HookRegistry", **kwargs: Any) -> None:
        """Register hook to handle EventLoopFailureEvent for MaxTokensReachedException."""
        registry.add_callback(EventLoopFailureEvent, self._handle_max_tokens_reached)

    def _handle_max_tokens_reached(self, event: EventLoopFailureEvent) -> None:
        """Handle MaxTokensReachedException by cleaning up orphaned tool uses and allowing continuation."""
        if not isinstance(event.exception, MaxTokensReachedException):
            return

        logger.info("Handling MaxTokensReachedException - inspecting incomplete message for invalid tool uses")

        incomplete_message: Message = event.exception.incomplete_message

        if not incomplete_message["content"]:
            # Cannot correct invalid content block if content is empty
            return

        valid_content: list[ContentBlock] = []
        for content in incomplete_message["content"]:
            tool_use: ToolUse | None = content.get("toolUse")
            if not tool_use:
                valid_content.append(content)
                continue

            """
            Ideally this would be future proofed using a pydantic validator. Since ToolUse is not implemented
            using pydantic, we inspect each field.
            """
            # Check if tool use is incomplete (missing or empty required fields)
            tool_name = tool_use.get("name")
            if not (tool_name and tool_use.get("input") and tool_use.get("toolUseId")):
                """
                If tool_use does not conform to the expected schema it means the max_tokens issue resulted in it not 
                being populated it correctly.
                
                It is safe to drop the content block, but we insert a new one to ensure Agent is aware of failure
                on the next iteration.
                """
                display_name = tool_name if tool_name else "<unknown>"
                logger.warning(
                    "tool_name=<%s> | replacing with error message due to max_tokens truncation.", display_name
                )

                valid_content.append(
                    {
                        "text": f"The selected tool {display_name}'s tool use was incomplete due "
                        f"to maximum token limits being reached."
                    }
                )
            else:
                # ToolUse was invalid for an unknown reason. Cannot correct, return and allow exception to propagate up.
                return

        valid_message: Message = {"content": valid_content, "role": incomplete_message["role"]}
        event.agent.messages.append(valid_message)
        event.agent.hooks.invoke_callbacks(MessageAddedEvent(agent=event.agent, message=valid_message))
        event.should_continue_loop = True
