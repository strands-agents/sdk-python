"""ContextOffloader plugin for managing large tool outputs.

This module provides the ContextOffloader plugin that intercepts oversized
tool results, persists each content block to a storage backend, and replaces
the in-context result with a truncated preview and per-block references.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.context_offloader import (
        ContextOffloader,
        InMemoryStorage,
        FileStorage,
    )

    # In-memory storage
    agent = Agent(plugins=[
        ContextOffloader(storage=InMemoryStorage())
    ])

    # File storage with custom thresholds and retrieval tool enabled
    agent = Agent(plugins=[
        ContextOffloader(
            storage=FileStorage("./artifacts"),
            max_result_tokens=5_000,
            preview_tokens=2_000,
            include_retrieval_tool=True,
        )
    ])
    ```
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ...hooks.events import AfterToolCallEvent
from ...models.model import _get_encoding
from ...plugins import Plugin, hook
from ...tools.decorator import tool
from ...types.content import Message
from ...types.tools import ToolContext, ToolResult, ToolResultContent
from .storage import Storage

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULT_TOKENS = 2_500
"""Default token threshold above which tool results are offloaded."""

_DEFAULT_PREVIEW_TOKENS = 1_000
"""Default number of tokens to keep as a preview in context."""

_CHARS_PER_TOKEN = 4
"""Approximate characters per token, fallback for preview slicing without tiktoken."""


class ContextOffloader(Plugin):
    """Plugin that offloads oversized tool results to reduce context consumption.

    When a tool result exceeds the configured token threshold, this plugin
    stores each content block individually to a storage backend and replaces
    the in-context result with a truncated text preview plus per-block references.

    Token estimation uses the agent's model ``count_tokens`` method, which
    leverages tiktoken when available and falls back to character-based heuristics.

    Content type handling:

    - **Text**: stored as ``text/plain``, replaced with a preview
    - **JSON**: stored as ``application/json``, replaced with a preview
    - **Image**: stored in its native format (e.g., ``image/png``), replaced with a
      placeholder showing format and size
    - **Document**: stored in its native format (e.g., ``application/pdf``), replaced
      with a placeholder showing format, name, and size
    - **Unknown types**: passed through unchanged

    This operates proactively at tool execution time via ``AfterToolCallEvent``,
    before the result enters the conversation — unlike ``SlidingWindowConversationManager``
    which truncates reactively after context overflow.

    Args:
        storage: Backend for storing offloaded content (required).
        max_result_tokens: Offload results whose estimated token count exceeds this threshold.
        preview_tokens: Number of tokens to keep as a text preview in context.
        include_retrieval_tool: Whether to register the ``retrieve_offloaded_content`` tool.
            Defaults to False.

    Example:
        ```python
        from strands import Agent
        from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage

        agent = Agent(plugins=[
            ContextOffloader(storage=InMemoryStorage())
        ])
        ```
    """

    name = "context_offloader"

    def __init__(
        self,
        storage: Storage,
        max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
        preview_tokens: int = _DEFAULT_PREVIEW_TOKENS,
        *,
        include_retrieval_tool: bool = False,
    ) -> None:
        """Initialize the ContextOffloader plugin.

        Args:
            storage: Backend for storing offloaded content.
            max_result_tokens: Offload results whose estimated token count exceeds this
                threshold. Defaults to ``_DEFAULT_MAX_RESULT_TOKENS`` (2,500).
            preview_tokens: Number of tokens to keep as a text preview in context.
                Uses tiktoken for exact slicing when available, falls back to
                chars/4 heuristic. Defaults to ``_DEFAULT_PREVIEW_TOKENS`` (1,000).
            include_retrieval_tool: Whether to register the ``retrieve_offloaded_content``
                tool so the agent can fetch offloaded content. Defaults to False.

        Raises:
            ValueError: If max_result_tokens is not positive, preview_tokens is negative,
                or preview_tokens >= max_result_tokens.
        """
        if max_result_tokens <= 0:
            raise ValueError("max_result_tokens must be positive")
        if preview_tokens < 0:
            raise ValueError("preview_tokens must be non-negative")
        if preview_tokens >= max_result_tokens:
            raise ValueError("preview_tokens must be less than max_result_tokens")

        self._storage = storage
        self._max_result_tokens = max_result_tokens
        self._preview_tokens = preview_tokens
        self._include_retrieval_tool = include_retrieval_tool
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Conditionally register the retrieval tool."""
        if not self._include_retrieval_tool:
            # Remove the auto-discovered retrieval tool
            self._tools = [t for t in self._tools if t.tool_name != "retrieve_offloaded_content"]

    @tool(context=True)
    def retrieve_offloaded_content(
        self,
        reference: str,
        tool_context: ToolContext,
    ) -> dict | str:
        """Retrieve offloaded content by reference.

        Use this tool when you see a placeholder with a reference (ref: ...)
        and need the full content.

        Args:
            reference: The reference string from the offload placeholder.
            tool_context: Injected by the framework. Not user-facing.
        """
        try:
            content_bytes, content_type = self._storage.retrieve(reference)
        except KeyError:
            return f"Error: reference not found: {reference}"

        if content_type.startswith("text/"):
            return content_bytes.decode("utf-8")

        if content_type == "application/json":
            return {"status": "success", "content": [{"json": json.loads(content_bytes)}]}

        if content_type.startswith("image/"):
            img_format = content_type.split("/")[-1]
            return {
                "status": "success",
                "content": [{"image": {"format": img_format, "source": {"bytes": content_bytes}}}],
            }

        if content_type.startswith("application/"):
            doc_format = content_type.split("/")[-1]
            doc_block = {"format": doc_format, "name": reference, "source": {"bytes": content_bytes}}
            return {"status": "success", "content": [{"document": doc_block}]}

        return content_bytes.decode("utf-8", errors="replace")

    @hook
    async def _handle_tool_result(self, event: AfterToolCallEvent) -> None:
        """Intercept oversized tool results, offload per-block, and replace with preview."""
        if event.cancel_message is not None:
            return

        if self._include_retrieval_tool and event.tool_use.get("name") == self.retrieve_offloaded_content.tool_name:
            return

        result = event.result
        content = result["content"]
        tool_use_id = event.tool_use["toolUseId"]

        # Estimate token count by wrapping the tool result as a message for count_tokens
        tool_result_message: Message = {"role": "user", "content": [{"toolResult": result}]}
        token_count = await event.agent.model.count_tokens([tool_result_message])

        if token_count <= self._max_result_tokens:
            return

        # Build text preview from text+JSON blocks.
        # Empty text blocks are intentionally excluded — they add no content value.
        text_preview_parts: list[str] = []
        for block in content:
            if block.get("text"):
                text_preview_parts.append(block["text"])
            elif "json" in block:
                text_preview_parts.append(json.dumps(block["json"], indent=2))

        full_text = "\n".join(text_preview_parts) if text_preview_parts else ""

        # Store each content block individually
        references: list[tuple[str, str, str]] = []  # (ref, content_type, description)
        try:
            for i, block in enumerate(content):
                key = f"{tool_use_id}_{i}"
                if block.get("text"):
                    ref = self._storage.store(key, block["text"].encode("utf-8"), "text/plain")
                    references.append((ref, "text/plain", f"text, {len(block['text']):,} chars"))
                elif "json" in block:
                    json_bytes = json.dumps(block["json"], indent=2).encode("utf-8")
                    ref = self._storage.store(key, json_bytes, "application/json")
                    references.append((ref, "application/json", f"json, {len(json_bytes):,} bytes"))
                elif "image" in block:
                    image = block["image"]
                    img_format = image.get("format", "unknown")
                    img_bytes = image.get("source", {}).get("bytes", b"")
                    if img_bytes:
                        ref = self._storage.store(key, img_bytes, f"image/{img_format}")
                        references.append((ref, f"image/{img_format}", f"image/{img_format}, {len(img_bytes):,} bytes"))
                    else:
                        references.append(("", f"image/{img_format}", f"image/{img_format}, 0 bytes"))
                elif "document" in block:
                    doc = block["document"]
                    doc_format = doc.get("format", "unknown")
                    doc_name = doc.get("name", "unknown")
                    doc_bytes = doc.get("source", {}).get("bytes", b"")
                    if doc_bytes:
                        ref = self._storage.store(key, doc_bytes, f"application/{doc_format}")
                        references.append((ref, f"application/{doc_format}", f"{doc_name}, {len(doc_bytes):,} bytes"))
                    else:
                        references.append(("", f"application/{doc_format}", f"{doc_name}, 0 bytes"))
        except Exception:
            logger.warning(
                "tool_use_id=<%s> | failed to offload tool result, keeping original",
                tool_use_id,
                exc_info=True,
            )
            return

        logger.debug(
            "tool_use_id=<%s>, blocks=<%d>, tokens=<%d> | tool result offloaded",
            tool_use_id,
            len(references),
            token_count,
        )

        # Build preview text — use tiktoken for exact slicing when available
        preview = self._slice_preview(full_text) if full_text else ""
        ref_lines = "\n".join(f"  {ref} ({desc})" for ref, _, desc in references if ref)

        guidance = (
            "Tool result was offloaded to external storage due to size.\n"
            "Use the preview below to answer if possible.\n"
            "Use your available tools to selectively access the data you need."
        )
        if self._include_retrieval_tool:
            guidance += "\nYou can also use retrieve_offloaded_content with a reference to get the full content."

        preview_text = (
            f"[Offloaded: {len(content)} blocks, ~{token_count:,} tokens]\n"
            f"{guidance}\n\n"
            f"{preview}\n\n"
            f"[Stored references:]\n{ref_lines}"
        )

        # Build new content with preview + placeholders for non-text blocks
        new_content: list[ToolResultContent] = [ToolResultContent(text=preview_text)]
        for i, block in enumerate(content):
            ref = references[i][0] if i < len(references) else ""
            if "text" in block or "json" in block:
                continue
            elif "image" in block:
                image = block["image"]
                img_format = image.get("format", "unknown")
                img_bytes = image.get("source", {}).get("bytes", b"")
                placeholder = f"[image: {img_format}, {len(img_bytes) if img_bytes else 0} bytes"
                if ref:
                    placeholder += f" | ref: {ref}"
                placeholder += "]"
                new_content.append(ToolResultContent(text=placeholder))
            elif "document" in block:
                doc = block["document"]
                doc_format = doc.get("format", "unknown")
                doc_name = doc.get("name", "unknown")
                doc_bytes = doc.get("source", {}).get("bytes", b"")
                placeholder = f"[document: {doc_format}, {doc_name}, {len(doc_bytes) if doc_bytes else 0} bytes"
                if ref:
                    placeholder += f" | ref: {ref}"
                placeholder += "]"
                new_content.append(ToolResultContent(text=placeholder))
            else:
                new_content.append(block)

        event.result = ToolResult(
            toolUseId=result["toolUseId"],
            status=result["status"],
            content=new_content,
        )

    def _slice_preview(self, text: str) -> str:
        """Slice text to approximately preview_tokens.

        Uses tiktoken for exact token-level slicing when available,
        falls back to characters (tokens * 4) otherwise.

        Args:
            text: The full text to slice.

        Returns:
            The preview text.
        """
        encoding = _get_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)
            preview: str = encoding.decode(tokens[: self._preview_tokens])
            return preview
        return text[: self._preview_tokens * _CHARS_PER_TOKEN]
