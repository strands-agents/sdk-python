"""A :class:`~strands.memory.types.MemoryStore` that persists its records through a storage backend.

A zero-infrastructure store for prototyping and testing. It defaults to an ephemeral in-memory
backend; pass a persistent :class:`~strands.storage.Storage` to keep memories across restarts.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from typing_extensions import Unpack

from ...memory.types import MemoryEntry, MemoryStore, Metadata, SearchOptions
from ...storage import InMemoryStorage, Storage
from ...storage.storage import _NAMESPACED, _NamespacedStorage
from .types import TestMemoryAddResult, TestMemoryStoreConfig

DEFAULT_MAX_SEARCH_RESULTS = 10

# Synthetic metadata key holding the token-overlap relevance score on a search result.
RELEVANCE_SCORE_KEY = "_relevanceScore"


def _new_id() -> str:
    """Return a fresh record identifier."""
    return str(uuid.uuid4())


def _now() -> str:
    """Return the current UTC time as a millisecond-precision, ``Z``-suffixed ISO 8601 string.

    This matches the format JavaScript's ``Date.prototype.toISOString()`` emits, so a record written
    by either SDK carries the same timestamp shape.
    """
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H:%M:%S')}.{now.microsecond // 1000:03d}Z"


def _sanitize_name(name: str) -> str:
    r"""Sanitize a store name into a safe single storage-key segment.

    Collapses parent-directory and separator sequences, then replaces any remaining unsafe
    character, guarding against a name that would escape the ``memory/`` prefix. Ensures cross-SDK
    compatibility.
    """
    sanitized = name.replace("..", "_").replace("/", "_").replace("\\", "_")
    return re.sub(r"[^\w\-.]", "_", sanitized, flags=re.ASCII)


def _tokenize(text: str) -> set[str]:
    r"""Lowercase and split text into a set of word tokens, dropping empties.

    Splits on any run of non-word characters. Ensures cross-SDK compatibility.
    """
    return {token for token in re.split(r"\W+", text.lower()) if token}


def _token_overlap_score(query_tokens: set[str], content: str) -> int:
    """Lexical relevance score for one record.

    The number of distinct query tokens that appear in the content; a higher count means more of the
    query's words are present. Returns 0 when there is no overlap.
    """
    return len(query_tokens & _tokenize(content))


class TestMemoryStore(MemoryStore):
    """A :class:`~strands.memory.types.MemoryStore` that persists its records through a storage backend.

    A zero-infrastructure store for prototyping and testing. The records are held as a single JSON
    blob under the key ``memory/<sanitized-store-name>.json`` within the :class:`~strands.storage.Storage`
    backend.

    Recall is lexical: results are ranked by how many query tokens overlap an entry's content, with
    the most recent entry winning ties. This is keyword matching, not the semantic search a managed
    vector store (e.g. :class:`~strands.vended_memory_stores.bedrock_knowledge_base.BedrockKnowledgeBaseStore`)
    provides.

    Each :meth:`add` rewrites the whole blob, so this fits modest volumes (hundreds to low thousands
    of entries), not production workloads — use a managed store like ``BedrockKnowledgeBaseStore`` for
    that. Writes within one event loop are serialized; concurrent writers across processes are not.

    The store defaults to an ephemeral :class:`~strands.storage.InMemoryStorage`: entries are lost
    when the process exits. Pass a persistent :class:`~strands.storage.Storage` (e.g.
    ``LocalFileStorage()``) to keep them across restarts.

    The serialized record format is shared with the TypeScript SDK's ``TestMemoryStore``: records use
    the same camelCase keys (``id``, ``content``, ``metadata``, ``createdAt``) and the same timestamp
    shape, so a backing store written by either SDK can be read by the other.

    Example:
        ```python
        from strands.vended_memory_stores.test_memory_store import TestMemoryStore
        from strands.storage import LocalFileStorage

        # Ephemeral by default; pass a LocalFileStorage to persist under ./.strands/memory/notes.json.
        store = TestMemoryStore(name="notes", storage=LocalFileStorage())

        result = await store.add("User prefers dark mode")
        results = await store.search("what theme does the user like?")
        ```
    """

    # Tell pytest not to collect this class as a test suite despite its ``Test`` prefix.
    __test__ = False

    def __init__(self, **store_config: Unpack[TestMemoryStoreConfig]) -> None:
        """Initialize the store.

        Args:
            **store_config: See :class:`TestMemoryStoreConfig`.

        Raises:
            ValueError: If ``name`` is empty/whitespace, or ``max_search_results`` is less than 1.
        """
        self.name = store_config["name"]
        if not self.name.strip():
            raise ValueError("TestMemoryStore: name must not be empty.")
        self.description = store_config.get("description")
        max_search_results = store_config.get("max_search_results")
        if max_search_results is not None and max_search_results < 1:
            raise ValueError("TestMemoryStore: max_search_results must be at least 1.")
        self.max_search_results = max_search_results
        # A local store is writable by default: the point is a zero-setup store you can write to.
        self.writable = store_config.get("writable", True)
        self.extraction = store_config.get("extraction")

        # Ephemeral by default. Scope every backend under `memory/`, unless the caller already namespaced it.
        backend = store_config.get("storage")
        if backend is None:
            backend = InMemoryStorage()
        self._storage: Storage = (
            backend if getattr(backend, "_namespaced", None) is _NAMESPACED else _NamespacedStorage(backend, "memory")
        )
        self._key = f"{_sanitize_name(self.name)}.json"

        # Serializes the read-modify-write cycle of add so concurrent adds don't each read the same
        # snapshot and clobber one another (last-write-wins).
        self._lock = asyncio.Lock()

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Search stored entries for those whose content overlaps the query.

        Results are ranked by query-token overlap, with the most recent entry winning ties.

        Args:
            query: The search query text.
            options: Optional search configuration.

        Returns:
            Matching memory entries ordered by relevance. Each entry's ``metadata`` includes a
            reserved synthetic ``_relevanceScore`` key (the token-overlap count). An empty or
            token-less query returns no results.

        Raises:
            ValueError: If ``options.max_search_results`` is less than 1, or the backing blob is
                malformed (invalid JSON, not an array, or a record missing required string fields).
            StorageError: If the backend read fails.
        """
        caller_max = options.get("max_search_results") if options is not None else None
        if caller_max is not None and caller_max < 1:
            raise ValueError("TestMemoryStore: max_search_results must be at least 1.")
        limit = caller_max or self.max_search_results or DEFAULT_MAX_SEARCH_RESULTS

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        records = await self._read()

        scored: list[tuple[dict[str, Any], int]] = []
        for record in records:
            score = _token_overlap_score(query_tokens, record["content"])
            if score > 0:
                scored.append((record, score))

        scored.sort(key=lambda item: (item[1], item[0]["createdAt"]), reverse=True)

        entries: list[MemoryEntry] = []
        for record, score in scored[:limit]:
            metadata: Metadata = {**(record.get("metadata") or {}), RELEVANCE_SCORE_KEY: score}
            entries.append(MemoryEntry(content=record["content"], metadata=metadata))
        return entries

    async def add(self, content: str, metadata: Metadata | None = None) -> TestMemoryAddResult:
        """Add ``content`` (with optional ``metadata``) to the store.

        Identical content is deduplicated: a repeat write returns the existing record's id without
        storing a second copy, so the at-least-once retries that extraction may perform never
        accumulate duplicates.

        Args:
            content: The text content to store.
            metadata: Optional metadata to attach to the entry. The key ``_relevanceScore`` is
                reserved: :meth:`search` populates it on results, so a value stored under it here is
                overwritten in search output.

        Returns:
            The id of the stored (or already-present) record.

        Raises:
            ValueError: If the store is not writable, ``content`` is empty/whitespace, or the
                existing backing blob is malformed.
            StorageError: If the backend read or write fails.
        """
        if not self.writable:
            raise ValueError("TestMemoryStore: store is not writable. Set writable=True in config to enable add().")
        if not content.strip():
            raise ValueError("TestMemoryStore: content must not be empty.")

        # The lock serializes the whole read-modify-write cycle so concurrent adds don't each read the
        # same snapshot and clobber one another. Reading inside the critical section guarantees add #N
        # sees add #N-1's write.
        async with self._lock:
            records = await self._read()

            normalized_content = content.strip()
            for record in records:
                if record["content"].strip() == normalized_content:
                    return TestMemoryAddResult(id=record["id"])

            new_record: dict[str, Any] = {"id": _new_id(), "content": content, "createdAt": _now()}
            if metadata is not None:
                new_record["metadata"] = metadata

            await self._write([*records, new_record])
            return TestMemoryAddResult(id=new_record["id"])

    async def _read(self) -> list[dict[str, Any]]:
        """Read and parse the record blob from storage; a missing key (or empty store) starts empty.

        Reads fresh on every call — there is no in-memory cache, so a search always reflects the
        latest write (including from another writer sharing the backend).

        Raises:
            ValueError: If the stored blob is not valid JSON, is not an array, or holds a record
                missing the required string fields.
            StorageError: If the backend read fails.
        """
        data = await self._storage.read(self._key)
        if data is None:
            return []

        try:
            parsed_blob = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(f"TestMemoryStore: invalid JSON in {self._key}: {error}") from error

        if not isinstance(parsed_blob, list):
            raise ValueError(f"TestMemoryStore: invalid backing store {self._key}: expected a JSON array of records")
        for record in parsed_blob:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not isinstance(record.get("content"), str)
                or not isinstance(record.get("createdAt"), str)
            ):
                raise ValueError(
                    f"TestMemoryStore: invalid backing store {self._key}: "
                    "each record must have string 'id', 'content', and 'createdAt' fields"
                )
            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    f"TestMemoryStore: invalid backing store {self._key}: "
                    "a record's 'metadata', when present, must be a JSON object"
                )
        return parsed_blob

    async def _write(self, records: list[dict[str, Any]]) -> None:
        """Persist ``records`` as a single JSON blob through the storage backend.

        Callers serialize invocations via the instance lock; atomicity is the backend's
        responsibility. A backend I/O failure surfaces as its own ``StorageError``, naming the key.
        """
        data = json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8")
        await self._storage.write(self._key, data)
