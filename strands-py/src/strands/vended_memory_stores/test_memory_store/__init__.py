"""A :class:`~strands.memory.types.MemoryStore` that persists its records through a storage backend.

A zero-infrastructure store for prototyping and offline use: no cloud account or provisioned
resources required. Ephemeral by default; pass a persistent :class:`~strands.storage.Storage` to
keep memories across restarts.

Example:
    ```python
    from strands.vended_memory_stores.test_memory_store import TestMemoryStore
    from strands.storage import LocalFileStorage

    store = TestMemoryStore(name="notes", storage=LocalFileStorage())
    ```
"""

from .store import TestMemoryStore
from .types import TestMemoryAddResult, TestMemoryStoreConfig

__all__ = [
    "TestMemoryAddResult",
    "TestMemoryStore",
    "TestMemoryStoreConfig",
]
