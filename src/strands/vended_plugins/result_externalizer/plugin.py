"""ToolResultExternalizer plugin for managing large tool outputs.

This module provides the ToolResultExternalizer plugin that intercepts oversized
tool results, persists the full content to a storage backend, and replaces the
in-context result with a truncated preview and reference.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.result_externalizer import (
        ToolResultExternalizer,
        InMemoryExternalizationStorage,
        FileExternalizationStorage,
    )

    # In-memory storage
    agent = Agent(plugins=[
        ToolResultExternalizer(storage=InMemoryExternalizationStorage())
    ])

    # File storage with custom thresholds
    agent = Agent(plugins=[
        ToolResultExternalizer(
            storage=FileExternalizationStorage("./artifacts"),
            max_result_chars=20_000,
            preview_chars=8_000,
        )
    ])
    ```
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...hooks.events import AfterToolCallEvent
from ...plugins import Plugin, hook
from ...types.tools import ToolResult, ToolResultContent
from .storage import ExternalizationStorage

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULT_CHARS = 10_000
"""Default character threshold above which tool results are externalized."""

_DEFAULT_PREVIEW_CHARS = 4_000
"""Default number of characters to keep as a preview in context."""


class ToolResultExternalizer(Plugin):
    """Plugin that externalizes oversized tool results to reduce context consumption.

    When a tool result exceeds the configured character threshold, this plugin:

    1. Persists the full text and JSON content to a storage backend
    2. Replaces the in-context result with a truncated preview plus a reference
    3. Replaces image and document blocks with descriptive placeholders (following the
       ``SlidingWindowConversationManager`` pattern for images)

    This operates proactively at tool execution time via ``AfterToolCallEvent``,
    before the result enters the conversation — unlike ``SlidingWindowConversationManager``
    which truncates reactively after context overflow.

    Args:
        storage: Backend for storing externalized content (required).
        max_result_chars: Externalize text results exceeding this many characters.
        preview_chars: Number of characters to keep as a preview in context.

    Example:
        ```python
        from strands import Agent
        from strands.vended_plugins.result_externalizer import (
            ToolResultExternalizer,
            InMemoryExternalizationStorage,
        )

        agent = Agent(plugins=[
            ToolResultExternalizer(storage=InMemoryExternalizationStorage())
        ])
        ```
    """

    name = "result_externalizer"

    def __init__(
        self,
        storage: ExternalizationStorage,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        preview_chars: int = _DEFAULT_PREVIEW_CHARS,
    ) -> None:
        """Initialize the ToolResultExternalizer plugin.

        Args:
            storage: Backend for storing externalized content.
            max_result_chars: Externalize text results exceeding this many characters.
                Defaults to ``_DEFAULT_MAX_RESULT_CHARS`` (10,000).
            preview_chars: Number of characters to keep as a preview in context.
                Defaults to ``_DEFAULT_PREVIEW_CHARS`` (4,000).

        Raises:
            ValueError: If max_result_chars is not positive, preview_chars is negative,
                or preview_chars >= max_result_chars.
        """
        if max_result_chars <= 0:
            raise ValueError("max_result_chars must be positive")
        if preview_chars < 0:
            raise ValueError("preview_chars must be non-negative")
        if preview_chars >= max_result_chars:
            raise ValueError("preview_chars must be less than max_result_chars")

        self._storage = storage
        self._max_result_chars = max_result_chars
        self._preview_chars = preview_chars
        super().__init__()

    @hook
    def _handle_tool_result(self, event: AfterToolCallEvent) -> None:
        """Intercept oversized tool results, externalize, and replace with preview."""
        if event.cancel_message is not None:
            return

        result = event.result
        content = result["content"]

        # Collect externalizable content (text + JSON) and measure total size.
        # Empty text blocks are intentionally excluded — they add no content value.
        text_parts: list[str] = []
        for block in content:
            if block.get("text"):
                text_parts.append(block["text"])
            elif "json" in block:
                text_parts.append(json.dumps(block["json"], indent=2))

        if not text_parts:
            return

        full_text = "\n".join(text_parts)
        if len(full_text) <= self._max_result_chars:
            return

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

        logger.debug(
            "tool_use_id=<%s>, total_chars=<%d>, ref=<%s> | tool result externalized",
            event.tool_use.get("toolUseId"),
            len(full_text),
            reference,
        )

        # Build preview
        preview = full_text[: self._preview_chars]
        preview_text = (
            f"[Externalized: {len(full_text):,} chars | ref: {reference}]\n\n"
            f"{preview}\n\n"
            f"[Full output stored externally: {reference}]"
        )

        # Build new content:
        #   - Preview text block first
        #   - Images replaced with placeholders (following SlidingWindowConversationManager)
        #   - Documents replaced with placeholders
        #   - Text and JSON blocks are already captured in the externalized content
        new_content: list[ToolResultContent] = [ToolResultContent(text=preview_text)]
        for block in content:
            if "image" in block:
                new_content.append(ToolResultContent(text=self._image_placeholder(block["image"])))
            elif "document" in block:
                new_content.append(ToolResultContent(text=self._document_placeholder(block["document"])))

        event.result = ToolResult(
            toolUseId=result["toolUseId"],
            status=result["status"],
            content=new_content,
        )

    @staticmethod
    def _image_placeholder(image_block: Any) -> str:
        """Create a descriptive placeholder for an image block.

        Follows the same pattern as ``SlidingWindowConversationManager``.

        Args:
            image_block: The image content block.

        Returns:
            A placeholder string describing the image.
        """
        source: Any = image_block.get("source", {})
        media_type = image_block.get("format", "unknown")
        data = source.get("bytes", b"")
        return f"[image: {media_type}, {len(data) if data else 0} bytes]"

    @staticmethod
    def _document_placeholder(document_block: Any) -> str:
        """Create a descriptive placeholder for a document block.

        Args:
            document_block: The document content block.

        Returns:
            A placeholder string describing the document.
        """
        name = document_block.get("name", "unknown")
        doc_format = document_block.get("format", "unknown")
        source: Any = document_block.get("source", {})
        data = source.get("bytes", b"")
        return f"[document: {doc_format}, {name}, {len(data) if data else 0} bytes]"
