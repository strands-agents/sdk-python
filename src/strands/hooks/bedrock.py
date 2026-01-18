"""Bedrock-specific hooks for AWS Bedrock features.

This module provides hook implementations for AWS Bedrock-specific functionality,
such as automatic prompt caching management.
"""

import logging
from typing import Any

from . import HookProvider, HookRegistry
from .events import AfterModelCallEvent, BeforeModelCallEvent

logger = logging.getLogger(__name__)

# Cache point object for Bedrock prompt caching
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
CACHE_POINT_ITEM: dict[str, Any] = {"cachePoint": {"type": "default"}}


class PromptCachingHook(HookProvider):
    """Hook provider for automatic Bedrock prompt caching management.

    This hook automatically manages cache points for AWS Bedrock's prompt caching feature.
    It adds a cache point to the last message before model invocation and removes it
    after the invocation completes.

    AWS Bedrock supports up to 4 cache points per request. This hook adds one cache point
    to enable the "Simplified Cache Management" feature for Claude models, which automatically
    checks for cache hits at content block boundaries (looking back approximately 20 content
    blocks from the cache checkpoint).

    Important Considerations:
        - This hook adds a cache point to the last message's content array
        - Bedrock has a maximum of 4 cache points per request
        - Claude models require minimum token counts (e.g., 1,024 for Claude 3.7 Sonnet)
        - Cache TTL is 5 minutes from the last access

    Example:
        ```python
        from strands import Agent
        from strands.models import BedrockModel
        from strands.hooks.bedrock import PromptCachingHook

        model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
        agent = Agent(
            model=model,
            hooks=[PromptCachingHook()]
        )
        ```

    See Also:
        - AWS Bedrock Prompt Caching: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
        - Strands Agents Hooks: https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/hooks/
    """

    def register_hooks(self, registry: HookRegistry) -> None:
        """Register hook callbacks with the registry.

        Args:
            registry: The hook registry to register callbacks with.
        """
        registry.add_callback(BeforeModelCallEvent, self.on_invocation_start)
        registry.add_callback(AfterModelCallEvent, self.on_invocation_end)

    def on_invocation_start(self, event: BeforeModelCallEvent) -> None:
        """Add cache point before model invocation.

        This callback is triggered before the model is invoked. It adds a cache point
        to the last message's content array to enable prompt caching.

        Args:
            event: The before model call event containing the agent and its messages.

        Note:
            If the messages list is empty or the last message has no content array,
            this method logs a warning and returns without modifying the messages.
        """
        messages = event.agent.messages

        # Validate messages structure
        if not messages:
            logger.warning("Cannot add cache point: messages list is empty")
            return

        last_message = messages[-1]
        if "content" not in last_message:
            logger.warning(
                "Cannot add cache point: last message has no content field | role=%s",
                last_message.get("role", "unknown"),
            )
            return

        content = last_message["content"]
        if not isinstance(content, list):
            logger.warning(
                "Cannot add cache point: content is not a list | type=%s | role=%s",
                type(content).__name__,
                last_message.get("role", "unknown"),
            )
            return

        # Add cache point to the end of the last message's content
        content.append(CACHE_POINT_ITEM)
        logger.debug(
            "Added cache point to message | message_index=%d | role=%s | content_blocks=%d",
            len(messages) - 1,
            last_message.get("role", "unknown"),
            len(content),
        )

    def on_invocation_end(self, event: AfterModelCallEvent) -> None:
        """Remove cache point after model invocation.

        This callback is triggered after the model invocation completes. It removes
        the cache point that was added in on_invocation_start to keep the message
        history clean.

        Args:
            event: The after model call event containing the agent and its messages.

        Note:
            If the cache point is not found in the last message's content array,
            this method logs a warning but does not raise an exception.
        """
        messages = event.agent.messages

        # Validate messages structure
        if not messages:
            logger.warning("Cannot remove cache point: messages list is empty")
            return

        last_message = messages[-1]
        if "content" not in last_message:
            logger.warning(
                "Cannot remove cache point: last message has no content field | role=%s",
                last_message.get("role", "unknown"),
            )
            return

        content = last_message["content"]
        if not isinstance(content, list):
            logger.warning(
                "Cannot remove cache point: content is not a list | type=%s | role=%s",
                type(content).__name__,
                last_message.get("role", "unknown"),
            )
            return

        # Remove cache point from the last message's content
        try:
            content.remove(CACHE_POINT_ITEM)
            logger.debug(
                "Removed cache point from message | message_index=%d | role=%s | content_blocks=%d",
                len(messages) - 1,
                last_message.get("role", "unknown"),
                len(content),
            )
        except ValueError:
            logger.warning(
                "Cache point not found in content | message_index=%d | role=%s | content_blocks=%d",
                len(messages) - 1,
                last_message.get("role", "unknown"),
                len(content),
            )
