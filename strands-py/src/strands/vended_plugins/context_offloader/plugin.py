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
import weakref
from typing import TYPE_CHECKING, TypeAlias, TypeGuard, cast

from typing_extensions import TypedDict

from ...hooks.events import AfterToolCallEvent, BeforeModelCallEvent
from ...plugins import Plugin, hook
from ...storage import Storage as UnifiedStorage
from ...tools.decorator import tool
from ...types.content import Message
from ...types.tools import ToolContext, ToolResult, ToolResultContent
from .search import _is_searchable_content, _search_content
from .storage import FileStorage, InMemoryStorage
from .storage import Storage as OffloaderStorage

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

# A storage backend accepted by ContextOffloader: either the SDK's unified
# ``strands.storage.Storage`` (write/read/delete/list) or a legacy offloader
# ``Storage`` (store/retrieve, deprecated).
_StorageBackend: TypeAlias = UnifiedStorage | OffloaderStorage

# Framed byte layout for a unified-Storage value: a 2-byte big-endian content-type
# length, the UTF-8 content-type, then the content bytes. This keeps the content-type
# metadata in the same key as the content (one key per block), mirroring strands-ts.
_CONTENT_TYPE_LENGTH_BYTES = 2


def _is_offloader_storage(storage: _StorageBackend) -> TypeGuard[OffloaderStorage]:
    """Return whether ``storage`` is a legacy offloader backend (has store/retrieve)."""
    return hasattr(storage, "store") and hasattr(storage, "retrieve")


def _frame_content(content: bytes, content_type: str) -> bytes:
    """Frame ``content`` with its ``content_type`` into a single unified-Storage value."""
    ct_bytes = content_type.encode("utf-8")
    return len(ct_bytes).to_bytes(_CONTENT_TYPE_LENGTH_BYTES, "big") + ct_bytes + content


def _unframe_content(frame: bytes) -> tuple[bytes, str]:
    """Split a framed unified-Storage value back into ``(content, content_type)``."""
    ct_len = int.from_bytes(frame[:_CONTENT_TYPE_LENGTH_BYTES], "big")
    start = _CONTENT_TYPE_LENGTH_BYTES
    content_type = frame[start : start + ct_len].decode("utf-8")
    content = frame[start + ct_len :]
    return content, content_type


async def _store_content(storage: _StorageBackend, key: str, content: bytes, content_type: str) -> str:
    """Store one content block and return its retrieval reference.

    Legacy backends persist the content-type natively via ``store()``; unified
    backends frame it into the stored bytes under ``key`` and use the key as the
    reference.
    """
    if _is_offloader_storage(storage):
        return await storage.store(key, content, content_type)
    await cast(UnifiedStorage, storage).write(key, _frame_content(content, content_type))
    return key


async def _retrieve_content(storage: _StorageBackend, reference: str) -> tuple[bytes, str]:
    """Retrieve one content block as ``(content, content_type)``.

    Raises:
        KeyError: If the reference is not found.
    """
    if _is_offloader_storage(storage):
        return await storage.retrieve(reference)
    data = await cast(UnifiedStorage, storage).read(reference)
    if data is None:
        raise KeyError(f"Reference not found: {reference}")
    return _unframe_content(data)


class LineRange(TypedDict):
    """A span of lines to retrieve (1-indexed, inclusive)."""

    start: int
    end: int


_DEFAULT_MAX_RESULT_TOKENS = 2_500
"""Default token threshold above which tool results are offloaded."""

_DEFAULT_PREVIEW_TOKENS = 1_000
"""Default number of tokens to keep as a preview in context."""

_CHARS_PER_TOKEN = 4
"""Approximate characters per token, fallback for preview slicing without tiktoken."""

_DEFAULT_EVICT_AFTER_CYCLES = 20
"""Default agent-loop cycles before an offloaded entry is evicted (unified Storage backends)."""


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
        storage: Backend for storing offloaded content (required). Accepts either a unified
            :class:`strands.storage.Storage` (``write``/``read``/``delete``/``list``; preferred)
            or a legacy offloader ``Storage`` (``store``/``retrieve``; deprecated) from this module.
        max_result_tokens: Offload results whose estimated token count exceeds this threshold.
        preview_tokens: Number of tokens to keep as a text preview in context.
        include_retrieval_tool: Whether to register the ``retrieve_offloaded_content`` tool.
            Defaults to True.
        evict_after_cycles: Agent loop cycles after which an offloaded entry is evicted. Applies
            to unified ``Storage`` backends (which have no built-in eviction). Defaults to 20;
            ``None`` disables eviction.

    Example:
        ```python
        from strands import Agent
        from strands.storage import InMemoryStorage

        agent = Agent(plugins=[
            ContextOffloader(storage=InMemoryStorage())
        ])
        ```
    """

    name = "context_offloader"

    def __init__(
        self,
        storage: _StorageBackend,
        max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
        preview_tokens: int = _DEFAULT_PREVIEW_TOKENS,
        *,
        include_retrieval_tool: bool = True,
        evict_after_cycles: int | None = _DEFAULT_EVICT_AFTER_CYCLES,
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
                tool so the agent can fetch offloaded content. Defaults to True.
            evict_after_cycles: Number of agent loop cycles after which an offloaded entry
                is evicted. Applies to unified ``Storage`` backends (which have no built-in
                eviction) and, when the backend is a legacy ``InMemoryStorage`` left at its
                default window, is forwarded to it. Defaults to 20. ``None`` disables eviction.
                For unified backends the window is measured from when an entry was *stored*
                (not last accessed), so a still-referenced entry is evicted once it ages out.

        Raises:
            ValueError: If max_result_tokens is not positive, preview_tokens is negative,
                preview_tokens >= max_result_tokens, or evict_after_cycles is not a
                positive integer or None.
        """
        if max_result_tokens <= 0:
            raise ValueError("max_result_tokens must be positive")
        if preview_tokens < 0:
            raise ValueError("preview_tokens must be non-negative")
        if preview_tokens >= max_result_tokens:
            raise ValueError("preview_tokens must be less than max_result_tokens")
        if evict_after_cycles is not None and (
            not isinstance(evict_after_cycles, int) or isinstance(evict_after_cycles, bool) or evict_after_cycles < 1
        ):
            raise ValueError("evict_after_cycles must be a positive integer or None")

        self._storage: _StorageBackend = storage
        # Per-agent FileStorage bound to that agent's sandbox; other backends are shared as-is.
        self._storage_by_agent: weakref.WeakKeyDictionary[Agent, _StorageBackend] = weakref.WeakKeyDictionary()
        self._max_result_tokens = max_result_tokens
        self._preview_tokens = preview_tokens
        self._include_retrieval_tool = include_retrieval_tool
        self._evict_after_cycles = evict_after_cycles
        # Cycle each unified-storage key was stored at, for plugin-driven eviction.
        # Unified Storage has no built-in eviction; a legacy InMemoryStorage evicts itself.
        self._key_stored_at: dict[str, int] = {}
        super().__init__()

    def _storage_for_agent(self, agent: Agent) -> _StorageBackend:
        """Return the storage for an agent, binding FileStorage to its sandbox.

        Non-FileStorage backends are shared across agents unchanged. A FileStorage
        is bound once per agent to that agent's sandbox via
        :meth:`FileStorage.for_sandbox`, so a shared plugin isolates artifacts per
        agent sandbox.

        Args:
            agent: The agent whose storage to resolve.

        Returns:
            The storage instance for this agent.
        """
        if not isinstance(self._storage, FileStorage):
            return self._storage
        storage = self._storage_by_agent.get(agent)
        if storage is None:
            storage = self._storage.for_sandbox(agent.sandbox)
            self._storage_by_agent[agent] = storage
        return storage

    def init_agent(self, agent: Agent) -> None:
        """Conditionally register the retrieval tool and bind storage."""
        if isinstance(self._storage, InMemoryStorage):
            self._storage._bind(id(agent), self._evict_after_cycles)
        # Bind FileStorage to this agent's sandbox up front (no-op for other backends).
        self._storage_for_agent(agent)
        if not self._include_retrieval_tool:
            # Remove the auto-discovered retrieval tool
            self._tools = [t for t in self._tools if t.tool_name != "retrieve_offloaded_content"]

    @hook
    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Trigger eviction of stale entries based on the agent's cycle count.

        A legacy ``InMemoryStorage`` evicts internally; unified ``Storage`` backends
        have no built-in eviction, so the plugin evicts their stale keys here.
        """
        cycle = event.agent.event_loop_metrics.cycle_count
        if isinstance(self._storage, InMemoryStorage):
            self._storage._evict(cycle)
        elif self._evict_after_cycles is not None:
            await self._evict(cycle)

    async def _evict(self, current_cycle: int) -> None:
        """Delete unified-storage entries stored more than ``evict_after_cycles`` cycles ago.

        Args:
            current_cycle: The agent's current event loop cycle count.
        """
        if self._evict_after_cycles is None:
            return
        threshold = current_cycle - self._evict_after_cycles
        to_evict = [key for key, stored_cycle in self._key_stored_at.items() if stored_cycle < threshold]
        if not to_evict:
            return
        storage = cast(UnifiedStorage, self._storage)
        for key in to_evict:
            # Stop tracking first, so a failed delete is not retried every cycle (matches strands-ts).
            del self._key_stored_at[key]
            try:
                await storage.delete(key)
            except Exception:
                logger.debug("key=<%s> | failed to evict offloaded entry", key, exc_info=True)
        logger.debug("evicted=<%d>, threshold=<%d> | evicted stale offloaded entries", len(to_evict), threshold)

    @tool(context=True)
    async def retrieve_offloaded_content(
        self,
        reference: str,
        tool_context: ToolContext,
        pattern: str | None = None,
        line_range: LineRange | None = None,
        context_lines: int | None = None,
    ) -> dict | str:
        """Retrieve offloaded content by reference.

        When a tool result was too large to keep in context, it was stored externally and replaced with a preview
        and a reference. Use this tool with that reference to access the stored content.

        Returns:
          - With pattern: matching lines with line numbers and surrounding context
          - With line_range: the specified span of lines with line numbers
          - Without pattern/line_range: the full original content (use sparingly — re-injects all tokens)

        Constraints:
          - pattern/line_range/context_lines only work on text content. For binary content, omit them.
          - Line numbers in results are 1-indexed and can be used in follow-up line_range calls.

        Examples:
          {"reference": "ref_1", "pattern": "error"} -> lines containing "error" with 5 lines context
          {"reference": "ref_1", "pattern": "error|warning", "context_lines": 3} -> regex, 3 lines context
          {"reference": "ref_1", "line_range": {"start": 10, "end": 25}} -> lines 10-25
          {"reference": "ref_1", "pattern": "TODO", "line_range": {"start": 1, "end": 50}} -> search within range

        Args:
            reference: The reference string from the offload placeholder (e.g. "mem_1_tool-123_0").
            pattern: Regex or keyword to grep for. Returns only matching lines with context — not the full content.
            line_range: Return only this span of lines. A dict with 'start' and 'end' keys (1-indexed).
                Combine with pattern to search within the range.
            context_lines: Lines before AND after each match (like grep -C). Default: 5.
                Without pattern/line_range, returns first N lines.
            tool_context: Injected by the framework. Not user-facing.
        """
        storage = self._storage_for_agent(tool_context.agent)
        try:
            content_bytes, content_type = await _retrieve_content(storage, reference)
        except KeyError:
            return f"Error: reference not found: {reference}"

        if pattern is None and line_range is None and context_lines is None:
            return self._decode_full_content(content_bytes, content_type, reference)

        if not _is_searchable_content(content_type):
            return (
                f"Error: cannot search binary content ({content_type}). "
                "Omit pattern/line_range/context_lines to retrieve the full content."
            )

        text = content_bytes.decode("utf-8")
        ctx_lines = context_lines if context_lines is not None else 5
        max_chars = self._max_result_tokens * _CHARS_PER_TOKEN

        lr: tuple[int, int] | None = None
        if line_range is not None:
            lr = (int(line_range["start"]), int(line_range["end"]))
        elif pattern is None:
            lr = (1, max(1, ctx_lines))

        return _search_content(text, pattern=pattern, line_range=lr, context_lines=ctx_lines, max_chars=max_chars)

    @staticmethod
    def _decode_full_content(content_bytes: bytes, content_type: str, reference: str) -> dict | str:
        """Decode stored content into its native format for full retrieval."""
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
        storage = self._storage_for_agent(event.agent)
        references: list[tuple[str, str, str]] = []  # (ref, content_type, description)
        try:
            for i, block in enumerate(content):
                key = f"{tool_use_id}_{i}"
                if block.get("text"):
                    ref = await _store_content(storage, key, block["text"].encode("utf-8"), "text/plain")
                    references.append((ref, "text/plain", f"text, {len(block['text']):,} chars"))
                elif "json" in block:
                    json_bytes = json.dumps(block["json"], indent=2).encode("utf-8")
                    ref = await _store_content(storage, key, json_bytes, "application/json")
                    references.append((ref, "application/json", f"json, {len(json_bytes):,} bytes"))
                elif "image" in block:
                    image = block["image"]
                    img_format = image.get("format", "unknown")
                    img_bytes = image.get("source", {}).get("bytes", b"")
                    if img_bytes:
                        ref = await _store_content(storage, key, img_bytes, f"image/{img_format}")
                        references.append((ref, f"image/{img_format}", f"image/{img_format}, {len(img_bytes):,} bytes"))
                    else:
                        references.append(("", f"image/{img_format}", f"image/{img_format}, 0 bytes"))
                elif "document" in block:
                    doc = block["document"]
                    doc_format = doc.get("format", "unknown")
                    doc_name = doc.get("name", "unknown")
                    doc_bytes = doc.get("source", {}).get("bytes", b"")
                    if doc_bytes:
                        ref = await _store_content(storage, key, doc_bytes, f"application/{doc_format}")
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

        # Track unified-storage keys so the plugin can evict them later (legacy backends
        # self-evict). Mirrors strands-ts, which records regardless of the eviction window.
        if not _is_offloader_storage(storage):
            stored_at = event.agent.event_loop_metrics.cycle_count
            for ref, _content_type, _description in references:
                if ref:
                    self._key_stored_at[ref] = stored_at

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
            "Use the preview below if it answers your question.\n"
        )
        if self._include_retrieval_tool:
            guidance += (
                "If you need more detail, use retrieve_offloaded_content with a reference and:\n"
                "  - pattern: regex or keyword to find matching lines with context\n"
                "  - line_range: { start, end } to read a specific span of lines\n"
                "Retrieve full content (omit pattern/line_range) as a last resort."
            )
        else:
            guidance += "If you need more detail, use your available tools to access specific data."

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
        """Slice text to approximately preview_tokens using character-based estimation.

        Args:
            text: The full text to slice.

        Returns:
            The preview text.
        """
        return text[: self._preview_tokens * _CHARS_PER_TOKEN]
