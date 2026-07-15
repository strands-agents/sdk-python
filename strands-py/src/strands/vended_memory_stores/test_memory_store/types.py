"""Configuration and result types for the JSON-blob memory store."""

from __future__ import annotations

from dataclasses import dataclass

from ...memory.types import MemoryStoreConfig
from ...storage import Storage


class TestMemoryStoreConfig(MemoryStoreConfig, total=False):
    """Full configuration for a :class:`TestMemoryStore`, passed as its constructor kwargs.

    Attributes:
        storage: Storage backend the records are persisted through. Records are held as a single JSON
            blob under the key ``memory/<sanitized-store-name>.json``. Defaults to an ephemeral
            :class:`~strands.storage.InMemoryStorage` — entries live only in memory and are lost when
            the process exits. Pass a ``LocalFileStorage`` (or any :class:`~strands.storage.Storage`)
            to persist across restarts, e.g. ``LocalFileStorage()`` to write under ``./.strands/``.
    """

    storage: Storage


# Tell pytest not to collect this class as a test suite despite its ``Test`` prefix. A TypedDict
# rejects a ``__test__`` entry in its body, so it is assigned after the class instead.
TestMemoryStoreConfig.__test__ = False  # type: ignore[attr-defined]


@dataclass
class TestMemoryAddResult:
    """Result returned by :meth:`TestMemoryStore.add`.

    Attributes:
        id: The generated id of the stored (or already-present, on dedup) record.
    """

    # Tell pytest not to collect this class as a test suite despite its ``Test`` prefix.
    __test__ = False

    id: str
