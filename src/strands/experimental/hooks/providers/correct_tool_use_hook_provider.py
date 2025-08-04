import logging
from typing import Any

from src.strands.hooks import MessageAddedEvent
from src.strands.types.tools import ToolUse
from strands.experimental.hooks.events import EventLoopFailureEvent
from strands.hooks import HookProvider, HookRegistry
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import MaxTokensReachedException

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
        valid_content: list[ContentBlock] = []

        for i, content in enumerate(incomplete_message["content"]):
            tool_use: ToolUse = content.get("toolUse")
            if not tool_use:
                valid_content.append(content)
                logger.debug(f"Content block {i}: Valid non-tool content preserved")
                continue

            """
            Ideally this would be future proofed using a pydantic validator. Since ToolUse is not implemented
            using pydantic, we inspect each field.
            """
            tool_name = tool_use.get("name", "<unknown>")
            tool_input = tool_use.get("input")
            tool_use_id = tool_use.get("toolUseId")

            if not (tool_name and tool_input and tool_use_id):
                """
                If tool_use does not conform to the expected schema it means the max_tokens issue resulted in it not 
                being populated it correctly.
                
                It is safe to drop the content block, but we insert a new one to ensure Agent is aware of failure
                on the next iteration.
                """
                logger.warning(
                    f"Invalid tool use found at content block {i}: tool_name='{tool_name}', "
                    f"Replacing with error message due to max_tokens truncation."
                )

                valid_content.append(
                    {
                        "text": f"The selected tool {tool_name}'s tool use was incomplete due "
                        f"to maximum token limits being reached."
                    }
                )
            else:
                # Tool use is invalid for an unknown reason. Cannot safely recover, so allow exception to propagate
                logger.debug(
                    f"Tool use at content block {i} appears complete but is still invalid. "
                    f"tool_name='{tool_name}', tool_use_id='{tool_use_id}'. "
                    f"Cannot safely recover - allowing exception to propagate."
                )
                return

        valid_message: Message = {"content": valid_content, "role": incomplete_message["role"]}
        event.agent.messages.append(valid_message)
        event.agent.hooks.invoke_callbacks(MessageAddedEvent(agent=event.agent, message=valid_message))
        event.should_continue_loop = True

        logger.info("MaxTokensReachedException handled successfully - continuing event loop")
