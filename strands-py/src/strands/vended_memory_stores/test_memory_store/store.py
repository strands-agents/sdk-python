"""A :class:`~strands.memory.types.MemoryStore` that persists to a local JSON file.

A zero-infrastructure store for prototyping and testing. It persists to disk by default so memories persist
across sessions, and can be set to ephemeral for testing.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import Unpack

from ...memory.types import MemoryEntry, MemoryStore, Metadata, SearchOptions
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
    r"""Sanitize a store name into a safe single-path-segment filename.

    Collapses parent-directory and separator sequences, then replaces any remaining unsafe
    character, guarding the default-path branch against a name that would escape the memory
    directory. Ensures cross-SDK compatibility.
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
    """A :class:`~strands.memory.types.MemoryStore` backed by an in-memory list and a local JSON file.

    A zero-infrastructure store for prototyping and testing. It persists to disk by default so memories persist
    across sessions. Set ``persist=False`` for an ephemeral, single-session store.

    Recall is lexical: results are ranked by how many query tokens overlap an entry's content, with
    the most recent entry winning ties. This is keyword matching, not the semantic search a managed
    vector store (e.g. :class:`~strands.vended_memory_stores.bedrock_knowledge_base.BedrockKnowledgeBaseStore`)
    provides.

    Each :meth:`add` rewrites the whole file, so this fits modest volumes (hundreds to low thousands
    of entries), not production workloads — use a managed store like ``BedrockKnowledgeBaseStore`` for
    that. Writes within a process are serialized; concurrent writers across processes are not.

    The on-disk format is shared with the TypeScript SDK's ``TestMemoryStore``: records use the same
    camelCase keys (``id``, ``content``, ``metadata``, ``createdAt``) and the same timestamp shape, so
    a backing file written by either SDK can be read by the other.

    Example:
        ```python
        from strands.vended_memory_stores.test_memory_store import TestMemoryStore

        # Persists to ~/.strands/memory/notes.json by default.
        store = TestMemoryStore(name="notes")

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
            ValueError: If ``name`` or ``path`` is empty/whitespace, or ``max_search_results`` is
                less than 1.
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

        self._persist = store_config.get("persist", True)
        path = store_config.get("path")
        if path is not None and not path.strip():
            raise ValueError("TestMemoryStore: path must not be empty.")
        if not self._persist:
            self._path: Path | None = None
        elif path is not None:
            self._path = Path(path)
        else:
            self._path = Path.home() / ".strands" / "memory" / f"{_sanitize_name(self.name)}.json"

        # Records load lazily on first read/write so construction never touches the filesystem.
        self._records: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

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
            ValueError: If ``options.max_search_results`` is less than 1.
        """
        caller_max = options.get("max_search_results") if options is not None else None
        if caller_max is not None and caller_max < 1:
            raise ValueError("TestMemoryStore: max_search_results must be at least 1.")
        limit = caller_max or self.max_search_results or DEFAULT_MAX_SEARCH_RESULTS

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        records = self._load()

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
            ValueError: If the store is not writable or ``content`` is empty/whitespace.
            OSError: If persisting the entry to disk fails (e.g. the path is unreachable or not
                writable), with the target path in the message.
        """
        if not self.writable:
            raise ValueError("TestMemoryStore: store is not writable. Set writable=True in config to enable add().")
        if not content.strip():
            raise ValueError("TestMemoryStore: content must not be empty.")

        # The lock serializes the whole load-modify-flush cycle so concurrent adds don't each load
        # the same snapshot and clobber one another (last-write-wins). Within a single event loop the
        # synchronous critical section is already atomic; the lock additionally guards a store shared
        # across OS threads.
        with self._lock:
            records = self._load()

            normalized_content = content.strip()
            for record in records:
                if record["content"].strip() == normalized_content:
                    return TestMemoryAddResult(id=record["id"])

            new_record: dict[str, Any] = {"id": _new_id(), "content": content, "createdAt": _now()}
            if metadata is not None:
                new_record["metadata"] = metadata

            # Flush the candidate list first and only commit it to the in-memory cache once the write
            # succeeds, so a failed flush never leaves a phantom record that later writes resurrect.
            next_records = [*records, new_record]
            self._flush(next_records)
            self._records = next_records
            return TestMemoryAddResult(id=new_record["id"])

    def _load(self) -> list[dict[str, Any]]:
        """Load records from disk on first use; ephemeral stores (and a missing file) start empty."""
        if self._records is not None:
            return self._records

        if self._path is None:
            self._records = []
            return self._records

        try:
            with open(self._path, encoding="utf-8") as file:
                parsed_file = json.load(file)
        except FileNotFoundError:
            self._records = []
            return self._records
        except json.JSONDecodeError as error:
            raise ValueError(f"TestMemoryStore: invalid JSON in {self._path}: {error}") from error
        except OSError as error:
            raise OSError(f"TestMemoryStore: failed to read {self._path}: {error}") from error

        if not isinstance(parsed_file, list):
            raise ValueError(f"TestMemoryStore: invalid backing file {self._path}: expected a JSON array of records")
        for record in parsed_file:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not isinstance(record.get("content"), str)
                or not isinstance(record.get("createdAt"), str)
            ):
                raise ValueError(
                    f"TestMemoryStore: invalid backing file {self._path}: "
                    "each record must have string 'id', 'content', and 'createdAt' fields"
                )
        self._records = parsed_file
        return self._records

    def _flush(self, records: list[dict[str, Any]]) -> None:
        """Persist ``records`` with an atomic write (write to a ``.tmp`` file, then replace).

        A crash mid-write can never leave a partially written file. A no-op for ephemeral stores.

        Raises:
            OSError: If the backing directory cannot be created or the file cannot be written
                (e.g. the path is unreachable or not writable), with the target path in the message.
        """
        if self._path is None:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_name(f"{self._path.name}.tmp")
            with open(tmp_path, "w", encoding="utf-8", newline="\n") as file:
                json.dump(records, file, indent=2, ensure_ascii=False)
            tmp_path.replace(self._path)
        except OSError as error:
            raise OSError(f"TestMemoryStore: failed to write {self._path}: {error}") from error
