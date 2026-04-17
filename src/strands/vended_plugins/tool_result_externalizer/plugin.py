"""ToolResultExternalizer plugin for managing large tool outputs.

This module provides the ToolResultExternalizer plugin that intercepts oversized
tool results, persists the full content to a storage backend, and replaces the
in-context result with a truncated preview and reference.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.tool_result_externalizer import (
        ToolResultExternalizer,
        FileExternalizationStorage,
    )

    # Simple — in-memory storage, default thresholds
    agent = Agent(plugins=[ToolResultExternalizer()])

    # Customized — file storage with custom thresholds
    agent = Agent(plugins=[
        ToolResultExternalizer(
            storage=FileExternalizationStorage("./artifacts"),
            size_threshold_chars=20_000,
            preview_chars=8_000,
        )
    ])
    ```
"""

from __future__ import annotations

import logging

from ...hooks.events import AfterToolCallEvent
from ...plugins import Plugin, hook
from ...types.tools import ToolResult, ToolResultContent
from .storage import ExternalizationStorage, InMemoryExternalizationStorage

logger = logging.getLogger(__name__)


class ToolResultExternalizer(Plugin):
    """Plugin that externalizes oversized tool results to reduce context consumption.

    When a tool result exceeds the configured character threshold, this plugin:

    1. Persists the full text content to a storage backend
    2. Replaces the in-context result with a truncated preview plus a reference
    3. Preserves any non-text content blocks (images, documents, JSON) as-is

    This operates proactively at tool execution time via ``AfterToolCallEvent``,
    before the result enters the conversation — unlike ``SlidingWindowConversationManager``
    which truncates reactively after context overflow.

    Args:
        storage: Backend for storing externalized content. Defaults to in-memory storage.
        size_threshold_chars: Externalize text results exceeding this many characters.
        preview_chars: Number of characters to keep as a preview in context.

    Example:
        ```python
        from strands import Agent
        from strands.vended_plugins.tool_result_externalizer import ToolResultExternalizer

        agent = Agent(plugins=[ToolResultExternalizer()])
        ```
    """

    name = "tool_result_externalizer"

    def __init__(
        self,
        storage: ExternalizationStorage | None = None,
        size_threshold_chars: int = 10_000,
        preview_chars: int = 4_000,
    ) -> None:
        """Initialize the ToolResultExternalizer plugin.

        Args:
            storage: Backend for storing externalized content. Defaults to in-memory storage.
            size_threshold_chars: Externalize text results exceeding this many characters.
            preview_chars: Number of characters to keep as a preview in context.
        """
        self._storage: ExternalizationStorage = storage or InMemoryExternalizationStorage()
        self._size_threshold_chars = size_threshold_chars
        self._preview_chars = preview_chars
        super().__init__()

    @hook
    def _handle_tool_result(self, event: AfterToolCallEvent) -> None:
        """Intercept oversized tool results, externalize, and replace with preview."""
        if event.cancel_message is not None:
            return

        result = event.result
        content = result["content"]

        # Collect text blocks and measure total size
        text_blocks: list[str] = []
        for block in content:
            text = block.get("text")
            if text:
                text_blocks.append(text)

        total_chars = sum(len(t) for t in text_blocks)
        if total_chars <= self._size_threshold_chars:
            return

        full_text = "\n".join(text_blocks)

        # Persist full content
        try:
            reference = self._storage.store(event.tool_use["toolUseId"], full_text)
        except Exception:
            logger.warning(
                "tool_use_id=<%s> | failed to externalize tool result, keeping original",
                event.tool_use.get("toolUseId"),
                exc_info=True,
            )
            return

        # Build preview
        preview = full_text[: self._preview_chars]
        preview_text = (
            f"[Externalized: {len(full_text):,} chars | ref: {reference}]\n\n"
            f"{preview}\n\n"
            f"[Full output stored externally: {reference}]"
        )

        # Build new content: preview text block + any non-text blocks from original
        new_content: list[ToolResultContent] = [ToolResultContent(text=preview_text)]
        for block in content:
            if "text" not in block:
                new_content.append(block)

        event.result = ToolResult(
            toolUseId=result["toolUseId"],
            status=result["status"],
            content=new_content,
        )
