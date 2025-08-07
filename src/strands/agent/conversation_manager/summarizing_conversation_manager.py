"""Summarizing conversation history management with configurable options."""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from typing_extensions import override

from ...types.content import Message
from ...types.exceptions import ContextWindowOverflowException
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
        summarization_system_prompt: Optional[str] = None,
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
        self._summary_message: Optional[Message] = None

    @override
    def restore_from_session(self, state: dict[str, Any]) -> Optional[list[Message]]:
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

    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None, **kwargs: Any) -> None:
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

    def _generate_summary(self, messages: List[Message], agent: "Agent") -> Message:
        """Generate a summary of the provided messages.

        Args:
            messages: The messages to summarize.
            agent: The agent instance to use for summarization.

        Returns:
            A message containing the conversation summary.

        Raises:
            Exception: If summary generation fails.
        """
        # Choose which agent to use for summarization
        summarization_agent = self.summarization_agent if self.summarization_agent is not None else agent

        # Save original system prompt and messages to restore later
        original_system_prompt = summarization_agent.system_prompt
        original_messages = summarization_agent.messages.copy()

        try:
            # Only override system prompt if no agent was provided during initialization
            if self.summarization_agent is None:
                # Use custom system prompt if provided, otherwise use default
                system_prompt = (
                    self.summarization_system_prompt
                    if self.summarization_system_prompt is not None
                    else DEFAULT_SUMMARIZATION_PROMPT
                )
                # Temporarily set the system prompt for summarization
                summarization_agent.system_prompt = system_prompt
            summarization_agent.messages = messages

            # Use the agent to generate summary with rich content (can use tools if needed)
            result = summarization_agent("Please summarize this conversation.")

            return result.message

        finally:
            # Restore original agent state
            summarization_agent.system_prompt = original_system_prompt
            summarization_agent.messages = original_messages

    def _adjust_split_point_for_tool_pairs(self, messages: List[Message], split_point: int) -> int:
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
