"""Token-aware conversation manager with LLM summarization.

Designed for autonomous agent workloads with long tool-call cycles. Uses actual input token count (from model
responses) to decide when to compact, and summarizes older context instead of just truncating.

Four-pass compaction strategy:
    1. Sanitize — strip ANSI escape codes, collapse repeated lines
    2. Truncate — replace oversized tool result content with a placeholder
    3. Summarize — use the LLM to summarize older messages (preserves context)
    4. Trim — remove oldest messages as last resort (loses context)
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ..._async import run_async
from ...event_loop.streaming import process_stream
from ...hooks import BeforeModelCallEvent, HookRegistry
from ...types.content import Message
from ...types.exceptions import ContextWindowOverflowException
from .conversation_manager import ConversationManager

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

# ANSI escape sequences: CSI codes, OSC sequences, charset designators, carriage returns
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\x1b\([A-Z]|\r")
_TOOL_RESULT_TRUNCATED = "The tool result was too large!"

SUMMARIZATION_PROMPT = (
    "You are a conversation summarizer for an autonomous AI agent. "
    "Create a concise summary preserving:\n"
    "- Current task/goal and progress\n"
    "- Key decisions made and reasoning\n"
    "- Important file paths, code changes, and tool results\n"
    "- Errors encountered and how they were resolved\n"
    "- Pending work items\n\n"
    "Format as bullet points. Be concise but don't lose critical context."
)


class TokenAwareConversationManager(ConversationManager):
    """Manages conversation based on token count with LLM summarization.

    Uses actual ``inputTokens`` from model responses to decide when to compact. Unlike
    ``SlidingWindowConversationManager`` which counts messages, this manager reacts to the real context size the model
    processes.

    The first user message (index 0) is always preserved across all compaction passes so the agent never loses sight of
    its original task.
    """

    def __init__(
        self,
        compact_threshold: int = 150_000,
        preserve_recent: int = 6,
        should_truncate_results: bool = True,
    ):
        """Initialize the token-aware conversation manager.

        Args:
            compact_threshold: Trigger compaction when inputTokens exceeds this value. Default 150 000 leaves ~50K
                headroom on a 200K context window.
            preserve_recent: Minimum number of recent messages to always keep.
            should_truncate_results: Replace oversized tool result content with a placeholder as a first reduction
                strategy.
        """
        super().__init__()
        self.compact_threshold = compact_threshold
        self.preserve_recent = preserve_recent
        self.should_truncate_results = should_truncate_results
        self._last_input_tokens: int = 0
        self._model_call_count: int = 0
        self._summary_message: Message | None = None

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks to track token usage and apply proactive management.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        super().register_hooks(registry, **kwargs)
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Proactive management: read token usage from the previous cycle and check budget.

        By the time this hook fires, ``start_cycle`` has already appended a new empty cycle to
        the invocation. The *previous* cycle (``cycles[-2]``) holds the most recent completed
        token counts. Reading ``cycles[-1]`` would always yield zero.

        Args:
            event: The before model call event.
        """
        self._model_call_count += 1

        # Read token count from the most recent *completed* cycle (the one before the current empty one)
        agent = event.agent
        invocation = agent.event_loop_metrics.latest_agent_invocation
        if invocation and len(invocation.cycles) >= 2:
            self._last_input_tokens = invocation.cycles[-2].usage.get("inputTokens", 0)

        if self._last_input_tokens > 0:
            self.apply_management(agent)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    @override
    def get_state(self) -> dict[str, Any]:
        """Return serialisable state for session persistence.

        Returns:
            Dictionary containing the manager's state.
        """
        state = super().get_state()
        state["last_input_tokens"] = self._last_input_tokens
        state["model_call_count"] = self._model_call_count
        state["summary_message"] = self._summary_message
        return state

    @override
    def restore_from_session(self, state: dict[str, Any]) -> list[Message] | None:
        """Restore manager state from a previous session.

        Args:
            state: Previous state of the conversation manager.

        Returns:
            Optionally returns the previous conversation summary if it exists.
        """
        result = super().restore_from_session(state)
        self._last_input_tokens = state.get("last_input_tokens", 0)
        self._model_call_count = state.get("model_call_count", 0)
        self._summary_message = state.get("summary_message")
        return [self._summary_message] if self._summary_message else result

    # ------------------------------------------------------------------
    # Core management interface
    # ------------------------------------------------------------------

    @override
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Proactively compact when token usage exceeds the threshold.

        Args:
            agent: The agent whose conversation history will be managed.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        if self._last_input_tokens <= self.compact_threshold:
            return

        logger.info(
            "input_tokens=<%d>, threshold=<%d>, message_count=<%d> | compacting conversation",
            self._last_input_tokens,
            self.compact_threshold,
            len(agent.messages),
        )
        self._compact(agent)

    @override
    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None:
        """Reactive reduction when a ``ContextWindowOverflowException`` is caught.

        Args:
            agent: The agent whose conversation history will be reduced.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        logger.warning("overflow=<true> | reduce_context triggered")
        self._compact(agent)

    # ------------------------------------------------------------------
    # Internal compaction logic
    # ------------------------------------------------------------------

    def _compact(self, agent: "Agent") -> None:
        """Run the four-pass compaction strategy.

        1. Sanitize all tool results (ANSI strip + dedup)
        2. Truncate oversized tool results (oldest first, skip first user message)
        3. Summarize older messages via LLM (preserve first user message)
        4. Hard-trim oldest messages as last resort (preserve first user message)

        The first user message (index 0) is always preserved — it contains the original task/prompt and must survive
        compaction so the agent never loses sight of what it was asked to do.

        Args:
            agent: The agent whose conversation history will be compacted.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
        """
        messages = agent.messages
        if len(messages) <= self.preserve_recent:
            raise ContextWindowOverflowException("Cannot reduce: at minimum message count")

        # The first message is the original user prompt — never touch it.
        protect_start = 1

        # Pass 1: sanitize all tool results
        self._sanitize_all_tool_results(messages)

        # Pass 2: truncate tool results (oldest first, skip protected + recent)
        if self.should_truncate_results:
            truncatable_end = len(messages) - self.preserve_recent
            truncated_count = 0
            for idx in range(protect_start, truncatable_end):
                if self._truncate_tool_results_in_message(messages, idx):
                    truncated_count += 1
            if truncated_count > 0:
                logger.info("truncated_count=<%d> | truncated tool results", truncated_count)
                return  # re-try with truncated results first

        # Pass 3: summarize older messages using the LLM
        summarize_end = len(messages) - self.preserve_recent
        messages_to_summarize_count = summarize_end - protect_start
        if messages_to_summarize_count > 0:
            split = self._adjust_split_for_tool_pairs(messages, summarize_end)
            if split > protect_start:
                try:
                    first_message = messages[0]
                    old_messages = messages[protect_start:split]
                    remaining = messages[split:]
                    summary = self._generate_summary(old_messages, agent)
                    self.removed_message_count += len(old_messages)
                    if self._summary_message:
                        self.removed_message_count -= 1
                    self._summary_message = summary
                    messages[:] = [first_message, summary] + remaining
                    logger.info(
                        "summarized_count=<%d>, remaining=<%d> | summarized older messages",
                        len(old_messages),
                        len(messages),
                    )
                    return
                except Exception as exc:
                    logger.warning("error=<%s> | summarization failed, falling back to trim", exc)

        # Pass 4: hard-trim as last resort (preserve first message)
        trim_target = max(self.preserve_recent, len(messages) // 2)
        trim_index = len(messages) - trim_target
        trim_index = max(trim_index, protect_start)
        trim_index = self._adjust_split_for_tool_pairs(messages, trim_index)
        if trim_index <= protect_start:
            raise ContextWindowOverflowException("Unable to trim conversation context!")

        first_message = messages[0]
        trimmed_count = trim_index - protect_start
        self.removed_message_count += trimmed_count
        messages[:] = [first_message] + messages[trim_index:]
        logger.info(
            "trimmed_count=<%d>, remaining=<%d> | trimmed oldest messages",
            trimmed_count,
            len(messages),
        )

    # ------------------------------------------------------------------
    # LLM summarization
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(old_messages: list[Message], agent: "Agent") -> Message:
        """Summarize older messages by calling the agent's model directly.

        Bypasses the full agent pipeline (lock, metrics, traces, tool loop) and simply asks the underlying model to
        summarize the conversation.

        Args:
            old_messages: The messages to summarize.
            agent: The parent agent whose model is used.

        Returns:
            A message containing the conversation summary with role ``assistant``.

        Raises:
            RuntimeError: If no response is received from the model.
        """
        summarization_messages: list[Message] = list(old_messages) + [
            {"role": "user", "content": [{"text": "Summarize this conversation concisely."}]}
        ]

        async def _call_model() -> Message:
            chunks = agent.model.stream(
                summarization_messages,
                tool_specs=None,
                system_prompt=SUMMARIZATION_PROMPT,
            )

            result_message: Message | None = None
            async for event in process_stream(chunks):
                if "stop" in event:
                    _, result_message, _, _ = event["stop"]

            if result_message is None:
                raise RuntimeError("Failed to generate summary: no response from model")
            return result_message

        message = run_async(_call_model)
        # Keep role as assistant — the summary sits between the preserved first user message
        # and the remaining conversation, maintaining proper user/assistant alternation.
        return message

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_all_tool_results(messages: list[Message]) -> None:
        """Strip ANSI codes and collapse repeated lines in all tool results.

        Args:
            messages: The full list of messages to sanitize in-place.
        """
        for msg in messages:
            for content in msg.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    for item in content["toolResult"].get("content", []):
                        text = item.get("text")
                        if text and ("\x1b" in text or "\r" in text):
                            item["text"] = _sanitize_text(text)

    @staticmethod
    def _truncate_tool_results_in_message(messages: list[Message], idx: int) -> bool:
        """Replace tool result content in a specific message with a placeholder.

        Args:
            messages: The full list of messages.
            idx: Index of the message to truncate.

        Returns:
            True if any tool results were truncated.
        """
        msg = messages[idx]
        changed = False
        for content in msg.get("content", []):
            if isinstance(content, dict) and "toolResult" in content:
                tr = content["toolResult"]
                for item in tr.get("content", []):
                    text = item.get("text", "")
                    if text and text != _TOOL_RESULT_TRUNCATED:
                        tr["status"] = "error"
                        tr["content"] = [{"text": _TOOL_RESULT_TRUNCATED}]
                        changed = True
                        break
        return changed

    @staticmethod
    def _adjust_split_for_tool_pairs(messages: list[Message], split: int) -> int:
        """Adjust split forward so it doesn't break toolUse/toolResult pairs.

        Args:
            messages: The full list of messages.
            split: The initially calculated split point.

        Returns:
            The adjusted split point.

        Raises:
            ContextWindowOverflowException: If no valid split point can be found.
        """
        while split < len(messages):
            if any("toolResult" in c for c in messages[split]["content"]) or (
                any("toolUse" in c for c in messages[split]["content"])
                and split + 1 < len(messages)
                and not any("toolResult" in c for c in messages[split + 1]["content"])
            ):
                split += 1
            else:
                break
        else:
            raise ContextWindowOverflowException("Unable to trim conversation context!")

        return split


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _sanitize_text(text: str) -> str:
    """Strip ANSI escape codes and collapse repeated consecutive lines.

    Args:
        text: Raw text potentially containing ANSI codes and repeated lines.

    Returns:
        Cleaned text with ANSI stripped and consecutive duplicate lines collapsed.
    """
    text = _ANSI_RE.sub("", text)
    lines = text.split("\n")
    result: list[str] = []
    prev: str | None = None
    repeat = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev and stripped:
            repeat += 1
        else:
            if repeat > 0:
                result.append(f"  [repeated {repeat} more time(s)]")
            result.append(line)
            prev = stripped
            repeat = 0
    if repeat > 0:
        result.append(f"  [repeated {repeat} more time(s)]")
    return "\n".join(result)
