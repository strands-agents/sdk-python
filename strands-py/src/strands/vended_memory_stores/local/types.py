"""Configuration and result types for the JSON-file memory store."""

from __future__ import annotations

from dataclasses import dataclass

from ...memory.types import MemoryStoreConfig


class JsonMemoryStoreConfig(MemoryStoreConfig, total=False):
    """Full configuration for a :class:`JsonMemoryStore`, passed as its constructor kwargs.

    The store persists to disk by default so memories persist across sessions.
    Set ``persist`` to ``False`` for an ephemeral, single-session store.

    Attributes:
        persist: Whether to persist entries to disk so they survive process restarts. ``True``
            (default) flushes writes to ``path`` (or the default location); ``False`` keeps entries
            in memory only, so they are lost when the process exits.
        path: Full path to the JSON file backing this store. Defaults to
            ``~/.strands/memory/<sanitized-store-name>.json``. Ignored when ``persist`` is ``False``.
    """

    persist: bool
    path: str


@dataclass
class JsonMemoryAddResult:
    """Result returned by :meth:`JsonMemoryStore.add`.

    Attributes:
        id: The generated id of the stored (or already-present, on dedup) record.
    """

    id: str
