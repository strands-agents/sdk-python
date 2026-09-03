"""Summarize strategy — replaces oversized content with LLM-generated summaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ....types.content import ContentBlock, Message
from ....types.tools import ToolResult, ToolResultContent
from ...methods.summarize import (
    SUMMARIZED_PREFIX,
    SummarizeConfig,
    flatten_messages_to_content,
    summarize_content,
    tool_result_to_content_blocks,
)
from ...types import ContextState
from .base import (
    BaseOffloadStrategy,
    OffloadConditions,
    OffloadTarget,
    repair_alternation,
    splice_with_pairs,
)

if TYPE_CHECKING:
    from ....agent.agent import Agent
    from ....models.model import Model

logger = logging.getLogger(__name__)


class SummarizeStrategy(BaseOffloadStrategy):
    """Summarize strategy — replaces oversized content with LLM-generated summaries."""

    @property
    def name(self) -> str:
        """Strategy name."""
        return "offload:summarize"

    def __init__(
        self,
        target: OffloadTarget | None = None,
        config: SummarizeConfig | None = None,
        conditions: OffloadConditions | None = None,
    ) -> None:
        super().__init__(target, conditions)
        self._config: SummarizeConfig = config or {}

    def when(self, **conditions: int | float) -> SummarizeStrategy:
        """Add conditions that determine when this strategy fires.

        Args:
            **conditions: Keyword arguments matching OffloadConditions fields.

        Returns:
            A new SummarizeStrategy instance with the conditions applied.
        """
        return SummarizeStrategy(self._target, self._config, OffloadConditions(**conditions))  # type: ignore[typeddict-item]

    async def apply(self, context: ContextState) -> bool:
        """Apply summarization. Returns False if no model is available.

        Args:
            context: Current context state.

        Returns:
            True if content was summarized.
        """
        if not self._resolve_model(context.agent):
            logger.warning("strategy=<%s> | no model available for summarization", self.name)
            return False
        return await super().apply(context)

    async def _apply_per_message(self, context: ContextState) -> bool:
        """Summarize oldest eligible messages into a single summary message."""
        model = self._resolve_model(context.agent)
        if not model:
            return False

        messages = context.messages
        if len(messages) <= 1:
            return False

        eligible = await self._get_eligible_messages(context)
        if not eligible:
            return False

        summarize_count = max(1, int(len(eligible) * self._removal_ratio))
        to_summarize = eligible[:summarize_count]

        content_blocks = flatten_messages_to_content(to_summarize)
        summary = await summarize_content(content_blocks, model, self._config)
        if not summary:
            return False

        total_tokens = await model.count_tokens(to_summarize)
        prefix = f"{SUMMARIZED_PREFIX} {len(to_summarize)} messages, ~{total_tokens:,} tokens]"
        summary_message = Message(
            role="user",
            content=[ContentBlock(text=f"{prefix}\n\n{summary}")],
        )

        removed, lowest_index = splice_with_pairs(messages, to_summarize)
        if removed == 0:
            return False

        insert_index = max(1, min(lowest_index, len(messages)))
        messages.insert(insert_index, summary_message)

        repair_alternation(messages)
        logger.debug("summarized=<%s>, tokens=<%s> | batched summarization complete", removed, total_tokens)
        return True

    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
    ) -> ContentBlock | None:
        model = self._resolve_model(agent)
        if not model:
            return None

        if "toolResult" in block:
            tool_result = block["toolResult"]
            content_blocks = tool_result_to_content_blocks(tool_result["content"])
            summary = await summarize_content(content_blocks, model, self._config)
            if not summary:
                return None

            logger.debug("tool_use_id=<%s>, tokens=<%s> | summarized tool result", tool_result["toolUseId"], tokens)
            text = f"{SUMMARIZED_PREFIX} ~{tokens:,} tokens]\n\n{summary}"
            summarized_content: list[ToolResultContent] = [{"text": text}]
            return ContentBlock(
                toolResult=ToolResult(
                    toolUseId=tool_result["toolUseId"],
                    status=tool_result["status"],
                    content=summarized_content,
                )
            )

        summary = await summarize_content([ContentBlock(text=block["text"])], model, self._config)
        if not summary:
            return None

        logger.debug("tracking_id=<%s>, tokens=<%s> | summarized text block", message.get("tracking_id"), tokens)
        return ContentBlock(text=f"{SUMMARIZED_PREFIX} ~{tokens:,} tokens]\n\n{summary}")

    def _resolve_model(self, agent: Agent) -> Model | None:
        return self._config.get("model") or agent.model
