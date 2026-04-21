"""Sliding window conversation history management."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...agent.agent import Agent

from ...hooks import BeforeModelCallEvent, HookRegistry
from ...types.content import ContentBlock, Messages
from ...types.exceptions import ContextWindowOverflowException
from ...types.tools import ToolResultContent
from ._token_utils import IMAGE_CHAR_ESTIMATE, TokenCounter, estimate_tokens
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

_PRESERVE_CHARS = 200


class SlidingWindowConversationManager(ConversationManager):
    """Implements a sliding window strategy for managing conversation history.

    This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids
    invalid window states.

    When truncation is enabled (the default), large tool results are partially truncated, preserving the first
    and last 200 characters, and image blocks inside tool results are replaced with descriptive text placeholders.
    Truncation targets the oldest tool results first so the most relevant recent context is preserved as long
    as possible.

    Supports proactive management during agent loop execution via the per_turn parameter.
    """

    def __init__(
        self,
        window_size: int = 40,
        should_truncate_results: bool = True,
        *,
        per_turn: bool | int = False,
        max_context_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
        compactable_after_messages: int | None = None,
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
            max_context_tokens: Optional maximum token budget for the conversation context.
                When set, the manager checks both message count and estimated token count,
                trimming oldest messages when the budget is exceeded. Uses the configured
                ``token_counter`` heuristic (chars/4 by default). Note: when both
                ``max_context_tokens`` and ``window_size`` are set, either limit can
                independently trigger context reduction.
            token_counter: Optional custom token counting function. Takes a Messages list
                and returns an integer token count. When not provided, the built-in
                ``estimate_tokens`` heuristic (chars/4) is used.
            compactable_after_messages: Optional message age after which tool results are
                replaced with a short stub (``[Tool result cleared — re-run if needed]``).
                This reclaims token budget from stale, re-runnable tool output while
                preserving the toolUse/toolResult pair structure required by model APIs.

        Raises:
            ValueError: If per_turn is 0 or a negative integer, or if compactable_after_messages
                is not a positive integer.
        """
        if isinstance(per_turn, int) and not isinstance(per_turn, bool) and per_turn <= 0:
            raise ValueError(f"per_turn must be a positive integer, True, or False, got {per_turn}")

        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be a positive integer, got {max_context_tokens}")

        if compactable_after_messages is not None and compactable_after_messages <= 0:
            raise ValueError(
                f"compactable_after_messages must be a positive integer, got {compactable_after_messages}"
            )

        super().__init__()

        self.window_size = window_size
        self.should_truncate_results = should_truncate_results
        self.per_turn = per_turn
        self.max_context_tokens = max_context_tokens
        self.token_counter: TokenCounter = token_counter or estimate_tokens
        self.compactable_after_messages = compactable_after_messages
        self._model_call_count = 0
        self._last_compacted_index = 0

    def register_hooks(self, registry: "HookRegistry", **kwargs: Any) -> None:
        """Register hook callbacks for per-turn conversation management.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().register_hooks(registry, **kwargs)

        # Always register — per_turn and max_context_tokens checks happen in the callback
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Handle before model call event for per-turn management and token budget enforcement.

        This callback is invoked before each model call. It applies management when either
        the token budget is exceeded or per-turn management is due. A single
        ``apply_management`` call handles both token budget and message count limits, so
        at most one call is made per hook invocation.

        Args:
            event: The before model call event containing the agent and model execution details.
        """
        needs_apply = False

        if self.max_context_tokens is not None:
            current_tokens = self._get_current_token_count(event.agent)
            if current_tokens > self.max_context_tokens:
                logger.debug(
                    "current_tokens=<%d>, max_context_tokens=<%d> | token budget exceeded",
                    current_tokens,
                    self.max_context_tokens,
                )
                needs_apply = True

        if self.per_turn is not False:
            self._model_call_count += 1

            if self.per_turn is True:
                needs_apply = True
            elif isinstance(self.per_turn, int) and self.per_turn > 0:
                if self._model_call_count % self.per_turn == 0:
                    needs_apply = True

        if needs_apply:
            logger.debug(
                "model_call_count=<%d>, per_turn=<%s> | applying conversation management",
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
        state["last_compacted_index"] = self._last_compacted_index
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
        self._last_compacted_index = state.get("last_compacted_index", 0)
        return result

    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply the sliding window to the agent's messages array to maintain a manageable history size.

        This method is called after every event loop cycle. It applies micro-compaction for stale tool
        results (if configured), then loops ``reduce_context`` until both message count and token budget
        limits are satisfied (or no further reduction is possible).

        Args:
            agent: The agent whose messages will be managed.
                This list is modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        messages = agent.messages

        # Micro-compact stale tool results before checking limits
        if self.compactable_after_messages is not None:
            self._micro_compact(messages)

        # Bound by len(messages) — each iteration must remove at least one message or
        # tool-result truncation, and the no-progress guard below catches stalls.
        max_iterations = len(messages)
        for _ in range(max_iterations):
            over_message_limit = len(messages) > self.window_size
            over_token_limit = (
                self.max_context_tokens is not None and self._get_current_token_count(agent) > self.max_context_tokens
            )

            if not over_message_limit and not over_token_limit:
                logger.debug(
                    "message_count=<%s>, window_size=<%s> | context within limits",
                    len(messages),
                    self.window_size,
                )
                return

            prev_len = len(messages)
            self.reduce_context(agent)
            if len(messages) >= prev_len:
                logger.warning(
                    "message_count=<%s>, window_size=<%s> | reduce_context made no progress, stopping",
                    len(messages),
                    self.window_size,
                )
                return

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
            ContextWindowOverflowException: If the context cannot be reduced further and a context overflow
                error was provided (e is not None). When called during routine window management (e is None),
                logs a warning and returns without modification.
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

        # Find the next valid trim point that:
        # 1. Starts with a user message (required by most model providers)
        # 2. Does not start with an orphaned toolResult
        # 3. Does not start with a toolUse unless its toolResult immediately follows
        while trim_index < len(messages):
            # Must start with a user message
            if messages[trim_index]["role"] != "user":
                trim_index += 1
                continue

            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any("toolResult" in content for content in messages[trim_index]["content"])
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    # Note: toolUse content normally appears only in assistant messages, but this
                    # check is kept as a defensive safeguard for non-standard message formats.
                    any("toolUse" in content for content in messages[trim_index]["content"])
                    and not (
                        trim_index + 1 < len(messages)
                        and any("toolResult" in content for content in messages[trim_index + 1]["content"])
                    )
                )
            ):
                trim_index += 1
            else:
                break
        else:
            # If we didn't find a valid trim_index
            if e is not None:
                raise ContextWindowOverflowException("Unable to trim conversation context!") from e
            logger.warning(
                "window_size=<%s>, message_count=<%s> | unable to trim conversation context, no valid trim point found",
                self.window_size,
                len(messages),
            )
            return

        # trim_index represents the number of messages being removed from the agents messages array
        self.removed_message_count += trim_index

        # Adjust compaction tracking index
        self._last_compacted_index = max(0, self._last_compacted_index - trim_index)

        # Overwrite message history
        messages[:] = messages[trim_index:]

    def _get_current_token_count(self, agent: "Agent") -> int:
        """Estimate the current token count for the conversation context.

        Always uses the configured ``token_counter`` heuristic rather than model-reported
        ``latest_context_size``, because the model-reported value reflects the *previous*
        cycle and becomes stale after any reduction — leading to over-reduction spirals.

        Args:
            agent: The agent whose context size is being measured.

        Returns:
            The estimated token count.
        """
        return self.token_counter(agent.messages)

    _COMPACT_STUB = "[Tool result cleared — re-run if needed]"

    def _micro_compact(self, messages: Messages) -> int:
        """Replace old tool results with compact stubs to reclaim token budget.

        Tool results older than ``compactable_after_messages`` messages from the end of the
        conversation are replaced with a short stub. The toolUse/toolResult pair structure
        is preserved — only the content within toolResult blocks is replaced.

        Tracks ``_last_compacted_index`` to skip already-processed messages on subsequent calls.

        Args:
            messages: The conversation message history (modified in-place).

        Returns:
            Estimated number of tokens reclaimed.
        """
        if self.compactable_after_messages is None:
            return 0

        # NOTE (M1): Clamp index in case messages were externally replaced or shortened
        # between calls (e.g., manual agent.messages reset, session restore mismatch).
        self._last_compacted_index = min(self._last_compacted_index, len(messages))

        reclaimed_chars = 0
        cutoff = len(messages) - self.compactable_after_messages

        # NOTE (M2): reclaimed_chars may overcount if text was already truncated by
        # _truncate_tool_results — the return value is an estimate, not used for decisions.
        stub_len = len(self._COMPACT_STUB)
        for i in range(self._last_compacted_index, max(0, cutoff)):
            msg = messages[i]
            for block in msg.get("content", []):
                if "toolResult" not in block:
                    continue
                result = block["toolResult"]
                items = result.get("content", [])
                for j, item in enumerate(items):
                    if "text" in item and item["text"] != self._COMPACT_STUB:
                        reclaimed_chars += max(0, len(item["text"]) - stub_len)
                        items[j] = {"text": self._COMPACT_STUB}
                    elif "image" in item:
                        reclaimed_chars += IMAGE_CHAR_ESTIMATE
                        items[j] = {"text": self._COMPACT_STUB}

        if cutoff > 0:
            self._last_compacted_index = max(self._last_compacted_index, cutoff)

        return reclaimed_chars // 4

    def _truncate_tool_results(self, messages: Messages, msg_idx: int) -> bool:
        """Truncate tool results and replace image blocks in a message to reduce context size.

        For text blocks within tool results, all blocks are partially truncated unless they
        have already been truncated. The first and last _PRESERVE_CHARS characters are kept,
        and the removed middle is replaced with a notice indicating how many characters were
        removed. The tool result status is not changed.

        Image blocks nested inside tool result content are replaced with a short descriptive placeholder.

        Args:
            messages: The conversation message history.
            msg_idx: Index of the message containing tool results to truncate.

        Returns:
            True if any changes were made to the message, False otherwise.
        """
        if msg_idx >= len(messages) or msg_idx < 0:
            return False

        def _image_placeholder(image_block: Any) -> str:
            source: Any = image_block.get("source", {})
            media_type = image_block.get("format", "unknown")
            data = source.get("bytes", b"")
            return f"[image: {media_type}, {len(data) if data else 0} bytes]"

        message = messages[msg_idx]
        changes_made = False
        new_content: list[ContentBlock] = []

        for content in message.get("content", []):
            if "toolResult" in content:
                tool_result: Any = content["toolResult"]
                tool_result_items = tool_result.get("content", [])
                new_items: list[ToolResultContent] = []
                item_changed = False

                for item in tool_result_items:
                    # Replace image items nested inside toolResult content
                    if "image" in item:
                        new_items.append({"text": _image_placeholder(item["image"])})
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
