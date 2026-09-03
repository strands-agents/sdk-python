"""Truncate strategy — replaces oversized content with a preview."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from ....types.content import ContentBlock, Message
from ...methods.truncate import TruncateConfig, truncate_text_block, truncate_tool_result
from .base import (
    BaseOffloadStrategy,
    OffloadConditions,
    OffloadTarget,
    repair_alternation,
    splice_with_pairs,
)

if TYPE_CHECKING:
    from ....agent.agent import Agent

from ...types import ContextState

logger = logging.getLogger(__name__)


class TruncateStrategy(BaseOffloadStrategy):
    """Truncate strategy — replaces oversized content with a head/tail preview."""

    @property
    def name(self) -> str:
        """Strategy name."""
        return "offload:truncate"

    def __init__(
        self,
        target: OffloadTarget | None = None,
        config: TruncateConfig | None = None,
        conditions: OffloadConditions | None = None,
    ) -> None:
        super().__init__(target, conditions)
        self._truncate_config: TruncateConfig = config or {}

        raw_preview = self._truncate_config.get("preview_tokens", 1000)
        preview_tokens = raw_preview if isinstance(raw_preview, (int, float)) and math.isfinite(raw_preview) else 1000

        if (
            conditions
            and "threshold" in conditions
            and isinstance(conditions["threshold"], (int, float))
            and conditions["threshold"] <= preview_tokens
        ):
            raise ValueError(
                f"threshold ({conditions['threshold']}) must be greater than preview_tokens ({preview_tokens}) "
                "to ensure truncation converges"
            )

    def when(self, **conditions: int | float) -> TruncateStrategy:
        """Add conditions that determine when this strategy fires.

        Args:
            **conditions: Keyword arguments matching OffloadConditions fields.

        Returns:
            A new TruncateStrategy instance with the conditions applied.
        """
        return TruncateStrategy(self._target, self._truncate_config, OffloadConditions(**conditions))  # type: ignore[typeddict-item]

    def _make_removal_marker(self, count: int) -> str:
        word = "message" if count == 1 else "messages"
        return f"[... {count} {word} elided ...]"

    async def _apply_per_message(self, context: ContextState) -> bool:
        """Message-level truncation: remove middle messages, keep head/tail."""
        messages = context.messages
        if len(messages) <= 1:
            return False

        eligible = await self._get_eligible_messages(context)
        if not eligible:
            return False

        preview_mode = self._truncate_config.get("preview", "head_tail")
        head_share = {"head": 1.0, "tail": 0.0, "head_tail": 0.3}[preview_mode]
        target_removal = max(1, int(len(eligible) * self._removal_ratio))
        keep_count = len(eligible) - target_removal

        head_keep = int(keep_count * head_share)
        tail_keep = keep_count - head_keep

        end_slice = len(eligible) - tail_keep if tail_keep > 0 else len(eligible)
        middle_messages = eligible[head_keep:end_slice]

        if not middle_messages:
            return False

        removed, lowest_index = splice_with_pairs(messages, middle_messages)
        if removed == 0:
            return False

        marker = self._make_removal_marker(removed)
        insert_index = max(1, min(lowest_index, len(messages)))
        messages.insert(insert_index, Message(role="user", content=[ContentBlock(text=marker)]))

        repair_alternation(messages)
        return True

    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
    ) -> ContentBlock | None:
        if "toolResult" in block:
            tool_use_id = block["toolResult"]["toolUseId"]
            logger.debug("tool_use_id=<%s>, tokens=<%s> | truncated tool result", tool_use_id, tokens)
            return ContentBlock(toolResult=truncate_tool_result(block["toolResult"], self._truncate_config))
        logger.debug(
            "tracking_id=<%s>, tokens=<%s> | truncated text block", message.get("tracking_id"), tokens
        )
        return truncate_text_block(block, self._truncate_config)
