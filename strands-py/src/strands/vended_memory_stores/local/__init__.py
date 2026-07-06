"""A :class:`~strands.memory.types.MemoryStore` that persists to a local JSON file.

A zero-infrastructure store for prototyping and offline use: no cloud account or provisioned
resources required. Persists to disk by default so an agent remembers across restarts.

Example:
    ```python
    from strands.vended_memory_stores.local import LocalMemoryStore

    store = LocalMemoryStore(name="notes")
    ```
"""

from .store import LocalMemoryStore
from .types import LocalMemoryAddResult, LocalMemoryStoreConfig

__all__ = [
    "LocalMemoryAddResult",
    "LocalMemoryStore",
    "LocalMemoryStoreConfig",
]
