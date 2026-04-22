"""ContextOffloader plugin for managing large tool outputs.

This module provides the ContextOffloader plugin that intercepts oversized
tool results, persists each content block to a storage backend, and replaces
the in-context result with a truncated preview and per-block references.
The plugin also provides a retrieval tool so the agent can fetch offloaded
content on demand.

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

    # File storage with custom thresholds
    agent = Agent(plugins=[
        ContextOffloader(
            storage=FileStorage("./artifacts"),
            max_result_chars=20_000,
            preview_chars=8_000,
        )
    ])
    ```
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ...hooks.events import AfterToolCallEvent, BeforeInvocationEvent
from ...plugins import Plugin, hook
from ...tools.decorator import tool
from ...types.tools import ToolContext, ToolResult, ToolResultContent
from .storage import Storage

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULT_CHARS = 10_000
"""Default character threshold above which tool results are offloaded."""

_DEFAULT_PREVIEW_CHARS = 4_000
"""Default number of characters to keep as a preview in context."""

_STATE_KEY = "context_offloader"

_SYSTEM_PROMPT_INJECTION = (
    "\n\n<context_offloader>\n"
    "When tool results are too large for the context window, they are offloaded to storage.\n"
    "You will see placeholders like [Offloaded: N blocks, M text chars] with references.\n"
    "To retrieve offloaded content, use the retrieve_offloaded_content tool with the reference.\n"
    "</context_offloader>"
)


class ContextOffloader(Plugin):
    """Plugin that offloads oversized tool results to reduce context consumption.

    When a tool result exceeds the configured character threshold, this plugin
    stores each content block individually to a storage backend and replaces
    the in-context result with a truncated text preview plus per-block references.
    A built-in retrieval tool allows the agent to fetch offloaded content on demand.

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
        max_result_chars: Offload results whose text+JSON content exceeds this many characters.
        preview_chars: Number of characters to keep as a text preview in context.

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
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
        preview_chars: int = _DEFAULT_PREVIEW_CHARS,
    ) -> None:
        """Initialize the ContextOffloader plugin.

        Args:
            storage: Backend for storing offloaded content.
            max_result_chars: Offload results whose text+JSON content exceeds this
                many characters. Defaults to ``_DEFAULT_MAX_RESULT_CHARS`` (10,000).
            preview_chars: Number of characters to keep as a text preview in context.
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

    @tool(context=True)
    def retrieve_offloaded_content(
        self,
        reference: str,
        tool_context: ToolContext,
    ) -> str:
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

        if content_type.startswith("text/") or content_type == "application/json":
            return content_bytes.decode("utf-8")

        return (
            f"[Binary content: {content_type}, {len(content_bytes):,} bytes]\n"
            f"Binary content cannot be displayed as text. "
            f"The content is stored at reference: {reference}"
        )

    @hook
    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Inject offloader guidance into the system prompt."""
        agent: Agent = event.agent
        state_data = agent.state.get(_STATE_KEY)
        last_injection = state_data.get("last_injection") if isinstance(state_data, dict) else None

        current_prompt = agent.system_prompt or ""

        if last_injection is not None and last_injection in current_prompt:
            return

        injection = _SYSTEM_PROMPT_INJECTION
        agent.system_prompt = f"{current_prompt}{injection}" if current_prompt else injection.strip()
        self._set_state_field(agent, "last_injection", injection)

    @hook
    def _handle_tool_result(self, event: AfterToolCallEvent) -> None:
        """Intercept oversized tool results, offload per-block, and replace with preview."""
        if event.cancel_message is not None:
            return

        result = event.result
        content = result["content"]
        tool_use_id = event.tool_use["toolUseId"]

        # Measure text+JSON size to decide whether to offload.
        # Empty text blocks are intentionally excluded — they add no content value.
        text_preview_parts: list[str] = []
        for block in content:
            if block.get("text"):
                text_preview_parts.append(block["text"])
            elif "json" in block:
                text_preview_parts.append(json.dumps(block["json"], indent=2))

        if not text_preview_parts:
            return

        full_text = "\n".join(text_preview_parts)
        if len(full_text) <= self._max_result_chars:
            return

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
            "tool_use_id=<%s>, blocks=<%d>, text_chars=<%d> | tool result offloaded",
            tool_use_id,
            len(references),
            len(full_text),
        )

        # Build preview text
        preview = full_text[: self._preview_chars]
        ref_lines = "\n".join(f"  {ref} ({desc})" for ref, _, desc in references if ref)
        preview_text = (
            f"[Offloaded: {len(content)} blocks, {len(full_text):,} text chars]\n\n"
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

    def _set_state_field(self, agent: Agent, key: str, value: str) -> None:
        """Set a field in the plugin's agent state dict.

        Args:
            agent: The agent whose state to update.
            key: The state field key.
            value: The value to set.
        """
        state_data = agent.state.get(_STATE_KEY)
        if not isinstance(state_data, dict):
            state_data = {}
        state_data[key] = value
        agent.state.set(_STATE_KEY, state_data)
