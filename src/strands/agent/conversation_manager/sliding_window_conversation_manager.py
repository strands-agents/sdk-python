"""Sliding window conversation history management."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...agent.agent import Agent

from ...hooks import BeforeModelCallEvent, HookRegistry
from ...types.content import ContentBlock, Messages
from ...types.exceptions import ContextWindowOverflowException
from ...types.tools import ToolResultContent
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

_PRESERVE_CHARS = 200


class SlidingWindowConversationManager(ConversationManager):
    """Implements a sliding window strategy for managing conversation history.

    This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids
    invalid window states.

    Supports proactive management during agent loop execution via the per_turn parameter.
    """

    def __init__(
        self,
        window_size: int = 40,
        should_truncate_results: bool = True,
        *,
        per_turn: bool | int = False,
    ):
        """Initialize the sliding window conversation manager.

        Args:
            window_size: Maximum number of messages to keep in the agent's history.
                Defaults to 40 messages.
            should_truncate_results: Truncate tool results when a message is too large for the model's context window
            per_turn: Controls when to apply message management during agent execution.
                - False (default): Only apply management at the end (default behavior)
                - True: Apply management before every model call
                - int (e.g., 3): Apply management before every N model calls

                When to use per_turn: If your agent performs many tool operations in loops
                (e.g., web browsing with frequent screenshots), enable per_turn to proactively
                manage message history and prevent the agent loop from slowing down. Start with
                per_turn=True and adjust to a specific frequency (e.g., per_turn=5) if needed
                for performance tuning.

        Raises:
            ValueError: If per_turn is 0 or a negative integer.
        """
        if isinstance(per_turn, int) and not isinstance(per_turn, bool) and per_turn <= 0:
            raise ValueError(f"per_turn must be a positive integer, True, or False, got {per_turn}")

        super().__init__()

        self.window_size = window_size
        self.should_truncate_results = should_truncate_results
        self.per_turn = per_turn
        self._model_call_count = 0

    def register_hooks(self, registry: "HookRegistry", **kwargs: Any) -> None:
        """Register hook callbacks for per-turn conversation management.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().register_hooks(registry, **kwargs)

        # Always register the callback - per_turn check happens in the callback
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Handle before model call event for per-turn management.

        This callback is invoked before each model call. It tracks the model call count and applies message management
        based on the per_turn configuration.

        Args:
            event: The before model call event containing the agent and model execution details.
        """
        # Check if per_turn is enabled
        if self.per_turn is False:
            return

        self._model_call_count += 1

        # Determine if we should apply management
        should_apply = False
        if self.per_turn is True:
            should_apply = True
        elif isinstance(self.per_turn, int) and self.per_turn > 0:
            should_apply = self._model_call_count % self.per_turn == 0

        if should_apply:
            logger.debug(
                "model_call_count=<%d>, per_turn=<%s> | applying per-turn conversation management",
                self._model_call_count,
                self.per_turn,
            )
            self.apply_management(event.agent)

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the conversation manager.

        Returns:
            Dictionary containing the manager's state, including model call count for per-turn tracking.
        """
        state = super().get_state()
        state["model_call_count"] = self._model_call_count
        return state

    def restore_from_session(self, state: dict[str, Any]) -> list | None:
        """Restore the conversation manager's state from a session.

        Args:
            state: Previous state of the conversation manager

        Returns:
            Optional list of messages to prepend to the agent's messages.
        """
        result = super().restore_from_session(state)
        self._model_call_count = state.get("model_call_count", 0)
        return result

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

    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None:
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
        oldest_message_idx_with_tool_results = self._find_oldest_message_with_tool_results(messages)
        if oldest_message_idx_with_tool_results is not None and self.should_truncate_results:
            logger.debug(
                "message_index=<%s> | found message with tool results at index", oldest_message_idx_with_tool_results
            )
            results_truncated = self._truncate_tool_results(messages, oldest_message_idx_with_tool_results)
            if results_truncated:
                logger.debug("message_index=<%s> | tool results truncated", oldest_message_idx_with_tool_results)
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
        """Truncate tool results and replace image blocks in a message to reduce context size.

        For text blocks within tool results, all blocks are partially truncated unless they
        have already been truncated. The first and last _PRESERVE_CHARS characters are kept,
        and the removed middle is replaced with a notice indicating how many characters were
        removed. The tool result status is not changed.

        Image blocks (both at the top level of the message content and nested inside tool
        result content) are replaced with a short descriptive placeholder.

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
        new_content: list[ContentBlock] = []

        for content in message.get("content", []):
            # Replace top-level image blocks with a descriptive placeholder
            if "image" in content:
                image: Any = content["image"]
                media_type = image.get("format", "unknown")
                source: Any = image.get("source", {})
                data = source.get("bytes", b"")
                size = len(data) if data else 0
                new_content.append({"text": f"[image: {media_type}, {size} bytes]"})
                changes_made = True
                continue

            if "toolResult" in content:
                tool_result: Any = content["toolResult"]
                tool_result_items = tool_result.get("content", [])
                new_items: list[ToolResultContent] = []
                item_changed = False

                for item in tool_result_items:
                    # Replace image items nested inside toolResult content
                    if "image" in item:
                        image = item["image"]
                        media_type = image.get("format", "unknown")
                        source = image.get("source", {})
                        data = source.get("bytes", b"")
                        size = len(data) if data else 0
                        new_items.append({"text": f"[image: {media_type}, {size} bytes]"})
                        item_changed = True
                        continue

                    # Partially truncate text items that have not already been truncated
                    if "text" in item:
                        text = item["text"]
                        truncation_marker = "... [truncated:"
                        if truncation_marker not in text and len(text) > 2 * _PRESERVE_CHARS:
                            prefix = text[:_PRESERVE_CHARS]
                            suffix = text[-_PRESERVE_CHARS:]
                            removed = len(text) - 2 * _PRESERVE_CHARS
                            truncated_text = (
                                f"{prefix}...\n\n... [truncated: {removed} chars removed] ...\n\n...{suffix}"
                            )
                            new_items.append({"text": truncated_text})
                            item_changed = True
                            continue

                    new_items.append(item)

                if item_changed:
                    updated_tool_result: Any = {
                        **{k: v for k, v in tool_result.items() if k != "content"},
                        "content": new_items,
                    }
                    new_content.append({"toolResult": updated_tool_result})
                    changes_made = True
                else:
                    new_content.append(content)
                continue

            new_content.append(content)

        if changes_made:
            message["content"] = new_content

        return changes_made

    def _find_oldest_message_with_tool_results(self, messages: Messages) -> int | None:
        """Find the index of the oldest message containing tool results.

        Iterates from oldest to newest so that truncation targets the least-recent
        (and therefore least relevant) tool results first.

        Args:
            messages: The conversation message history.

        Returns:
            Index of the oldest message with tool results, or None if no such message exists.
        """
        # Iterate from oldest to newest
        for idx in range(len(messages)):
            current_message = messages[idx]
            for content in current_message.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    return idx

        return None
