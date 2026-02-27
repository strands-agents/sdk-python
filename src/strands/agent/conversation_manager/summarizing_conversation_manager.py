"""Summarizing conversation history management with configurable options."""

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

from typing_extensions import override

from ..._async import run_async
from ...event_loop.streaming import process_stream
from ...tools._tool_helpers import noop_tool
from ...tools.registry import ToolRegistry
from ...types.content import ContentBlock, Message
from ...types.exceptions import ContextWindowOverflowException
from ...types.tools import AgentTool
from .conversation_manager import ConversationManager

if TYPE_CHECKING:
    from ..agent import Agent


logger = logging.getLogger(__name__)


DEFAULT_SUMMARIZATION_PROMPT = """You are a conversation summarizer. Provide a concise summary of the conversation \
history.

Format Requirements:
- You MUST create a structured and concise summary in bullet-point format.
- You MUST NOT respond conversationally.
- You MUST NOT address the user directly.
- You MUST NOT comment on tool availability.

Assumptions:
- You MUST NOT assume tool executions failed unless otherwise stated.

Task:
Your task is to create a structured summary document:
- It MUST contain bullet points with key topics and questions covered
- It MUST contain bullet points for all significant tools executed and their results
- It MUST contain bullet points for any code or technical information shared
- It MUST contain a section of key insights gained
- It MUST format the summary in the third person

Example format:

## Conversation Summary
* Topic 1: Key information
* Topic 2: Key information
*
## Tools Executed
* Tool X: Result Y"""


class SummarizingConversationManager(ConversationManager):
    """Implements a summarizing window manager.

    This manager provides a configurable option to summarize older context instead of
    simply trimming it, helping preserve important information while staying within
    context limits.
    """

    def __init__(
        self,
        summary_ratio: float = 0.3,
        preserve_recent_messages: int = 10,
        summarization_agent: Optional["Agent"] = None,
        summarization_system_prompt: str | None = None,
    ):
        """Initialize the summarizing conversation manager.

        Args:
            summary_ratio: Ratio of messages to summarize vs keep when context overflow occurs.
                Value between 0.1 and 0.8. Defaults to 0.3 (summarize 30% of oldest messages).
            preserve_recent_messages: Minimum number of recent messages to always keep.
                Defaults to 10 messages.
            summarization_agent: Optional agent to use for summarization instead of the parent agent.
                If provided, this agent can use tools as part of the summarization process.
            summarization_system_prompt: Optional system prompt override for summarization.
                If None, uses the default summarization prompt.
        """
        super().__init__()
        if summarization_agent is not None and summarization_system_prompt is not None:
            raise ValueError(
                "Cannot provide both summarization_agent and summarization_system_prompt. "
                "Agents come with their own system prompt."
            )

        self.summary_ratio = max(0.1, min(0.8, summary_ratio))
        self.preserve_recent_messages = preserve_recent_messages
        self.summarization_agent = summarization_agent
        self.summarization_system_prompt = summarization_system_prompt
        self._summary_message: Message | None = None

    @override
    def restore_from_session(self, state: dict[str, Any]) -> list[Message] | None:
        """Restores the Summarizing Conversation manager from its previous state in a session.

        Args:
            state: The previous state of the Summarizing Conversation Manager.

        Returns:
            Optionally returns the previous conversation summary if it exists.
        """
        super().restore_from_session(state)
        self._summary_message = state.get("summary_message")
        return [self._summary_message] if self._summary_message else None

    def get_state(self) -> dict[str, Any]:
        """Returns a dictionary representation of the state for the Summarizing Conversation Manager."""
        return {"summary_message": self._summary_message, **super().get_state()}

    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply management strategy to conversation history.

        For the summarizing conversation manager, no proactive management is performed.
        Summarization only occurs when there's a context overflow that triggers reduce_context.

        Args:
            agent: The agent whose conversation history will be managed.
                The agent's messages list is modified in-place.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # No proactive management - summarization only happens on context overflow
        pass

    def reduce_context(self, agent: "Agent", e: Exception | None = None, **kwargs: Any) -> None:
        """Reduce context using summarization.

        Args:
            agent: The agent whose conversation history will be reduced.
                The agent's messages list is modified in-place.
            e: The exception that triggered the context reduction, if any.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ContextWindowOverflowException: If the context cannot be summarized.
        """
        try:
            # Calculate how many messages to summarize
            messages_to_summarize_count = max(1, int(len(agent.messages) * self.summary_ratio))

            # Ensure we don't summarize recent messages
            messages_to_summarize_count = min(
                messages_to_summarize_count, len(agent.messages) - self.preserve_recent_messages
            )

            if messages_to_summarize_count <= 0:
                raise ContextWindowOverflowException("Cannot summarize: insufficient messages for summarization")

            # Adjust split point to avoid breaking ToolUse/ToolResult pairs
            messages_to_summarize_count = self._adjust_split_point_for_tool_pairs(
                agent.messages, messages_to_summarize_count
            )

            if messages_to_summarize_count <= 0:
                raise ContextWindowOverflowException("Cannot summarize: insufficient messages for summarization")

            # Extract messages to summarize
            messages_to_summarize = agent.messages[:messages_to_summarize_count]
            remaining_messages = agent.messages[messages_to_summarize_count:]

            # Keep track of the number of messages that have been summarized thus far.
            self.removed_message_count += len(messages_to_summarize)
            # If there is a summary message, don't count it in the removed_message_count.
            if self._summary_message:
                self.removed_message_count -= 1

            # Generate summary
            self._summary_message = self._generate_summary(messages_to_summarize, agent)

            # Replace the summarized messages with the summary
            agent.messages[:] = [self._summary_message] + remaining_messages

        except Exception as summarization_error:
            logger.error("Summarization failed: %s", summarization_error)
            raise summarization_error from e

    def _generate_summary(self, messages: list[Message], agent: "Agent") -> Message:
        """Generate a summary of the provided messages.

        When a dedicated summarization_agent was provided at init time, it is invoked as before
        (full agent pipeline, tool execution, etc.).

        In the default case (no summarization_agent), the parent agent's *model* is called
        directly via ``model.stream()``.  This avoids re-entering the agent pipeline which
        would deadlock on ``_invocation_lock`` and corrupt metrics / traces / interrupt state.

        Args:
            messages: The messages to summarize.
            agent: The agent instance whose model will be used for summarization when no
                dedicated summarization_agent was configured.

        Returns:
            A message containing the conversation summary.

        Raises:
            Exception: If summary generation fails.
        """
        if self.summarization_agent is not None:
            return self._generate_summary_with_agent(messages)

        return self._generate_summary_with_model(messages, agent)

    # ------------------------------------------------------------------
    # Path 1 – dedicated summarization agent (backward-compatible)
    # ------------------------------------------------------------------

    def _generate_summary_with_agent(self, messages: list[Message]) -> Message:
        """Generate a summary using the dedicated summarization agent.

        Args:
            messages: The messages to summarize.

        Returns:
            A message containing the conversation summary.
        """
        summarization_agent = self.summarization_agent
        assert summarization_agent is not None  # guaranteed by caller

        original_system_prompt = summarization_agent.system_prompt
        original_messages = summarization_agent.messages.copy()
        original_tool_registry = summarization_agent.tool_registry

        try:
            # Add no-op tool if agent has no tools to satisfy tool spec requirement
            if not summarization_agent.tool_names:
                tool_registry = ToolRegistry()
                tool_registry.register_tool(cast(AgentTool, noop_tool))
                summarization_agent.tool_registry = tool_registry

            summarization_agent.messages = messages

            result = summarization_agent("Please summarize this conversation.")

            # Filter content to only include blocks valid for user messages.
            # User messages cannot contain toolUse or reasoningContent blocks (those are assistant-only).
            # This is important when using structured_output_model, which adds toolUse blocks to responses.
            filtered_content: List[ContentBlock] = [
                content_block
                for content_block in result.message.get("content", [])
                if "toolUse" not in content_block and "reasoningContent" not in content_block
            ]

            # If no valid content remains after filtering, create a text block from the result
            if not filtered_content:
                # Use the string representation of the result as fallback
                filtered_content = [cast(ContentBlock, {"text": str(result)})]

            return cast(Message, {"role": "user", "content": filtered_content})

        finally:
            summarization_agent.system_prompt = original_system_prompt
            summarization_agent.messages = original_messages
            summarization_agent.tool_registry = original_tool_registry

    # ------------------------------------------------------------------
    # Path 2 – default case: call model.stream() directly
    # ------------------------------------------------------------------

    def _generate_summary_with_model(self, messages: list[Message], agent: "Agent") -> Message:
        """Generate a summary by calling the agent's model directly.

        This bypasses the full agent pipeline (lock, metrics, traces, tool loop) and
        simply asks the underlying model to summarize the conversation.

        Args:
            messages: The messages to summarize.
            agent: The parent agent whose model is used.

        Returns:
            A message containing the conversation summary.
        """
        system_prompt = (
            self.summarization_system_prompt
            if self.summarization_system_prompt is not None
            else DEFAULT_SUMMARIZATION_PROMPT
        )

        # Build the message list: conversation history + summarization request
        summarization_messages = list(messages) + [
            {"role": "user", "content": [{"text": "Please summarize this conversation."}]}
        ]

        async def _call_model() -> Message:
            chunks = agent.model.stream(
                summarization_messages,
                tool_specs=None,
                system_prompt=system_prompt,
            )

            result_message: Message | None = None
            async for event in process_stream(chunks):
                if "stop" in event:
                    _, result_message, _, _ = event["stop"]

            if result_message is None:
                raise RuntimeError("Failed to generate summary: no response from model")
            return result_message

        message = run_async(_call_model)
        return cast(Message, {**message, "role": "user"})

    def _adjust_split_point_for_tool_pairs(self, messages: list[Message], split_point: int) -> int:
        """Adjust the split point to avoid breaking ToolUse/ToolResult pairs.

        Uses the same logic as SlidingWindowConversationManager for consistency.

        Args:
            messages: The full list of messages.
            split_point: The initially calculated split point.

        Returns:
            The adjusted split point that doesn't break ToolUse/ToolResult pairs.

        Raises:
            ContextWindowOverflowException: If no valid split point can be found.
        """
        if split_point > len(messages):
            raise ContextWindowOverflowException("Split point exceeds message array length")

        if split_point == len(messages):
            return split_point

        # Find the next valid split_point
        while split_point < len(messages):
            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any("toolResult" in content for content in messages[split_point]["content"])
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    any("toolUse" in content for content in messages[split_point]["content"])
                    and split_point + 1 < len(messages)
                    and not any("toolResult" in content for content in messages[split_point + 1]["content"])
                )
            ):
                split_point += 1
            else:
                break
        else:
            # If we didn't find a valid split_point, then we throw
            raise ContextWindowOverflowException("Unable to trim conversation context!")

        return split_point
