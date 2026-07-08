"""Configuration and result types for the local memory store."""

from __future__ import annotations

from dataclasses import dataclass

from ...memory.types import MemoryStoreConfig


class LocalMemoryStoreConfig(MemoryStoreConfig, total=False):
    """Full configuration for a :class:`LocalMemoryStore`, passed as its constructor kwargs.

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
class LocalMemoryAddResult:
    """Result returned by :meth:`LocalMemoryStore.add`.

    Attributes:
        id: The generated id of the stored (or already-present, on dedup) record.
    """

    id: str
