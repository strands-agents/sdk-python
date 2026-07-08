"""A :class:`~strands.memory.types.MemoryStore` that persists to a local JSON file.

A zero-infrastructure store for prototyping and offline use: no cloud account or provisioned
resources required. Persists to disk by default so an agent remembers across restarts.

Example:
    ```python
    from strands.vended_memory_stores.local import JsonMemoryStore

    store = JsonMemoryStore(name="notes")
    ```
"""

import warnings
from typing import Any

from .store import JsonMemoryStore
from .types import JsonMemoryAddResult, JsonMemoryStoreConfig

__all__ = [
    "JsonMemoryAddResult",
    "JsonMemoryStore",
    "JsonMemoryStoreConfig",
]

# Deprecated aliases - warning emitted on access via __getattr__. Each maps its former name to the
# renamed symbol so old imports keep the same object identity.
_DEPRECATED_ALIASES = {
    "LocalMemoryStore": JsonMemoryStore,
    "LocalMemoryStoreConfig": JsonMemoryStoreConfig,
    "LocalMemoryAddResult": JsonMemoryAddResult,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        new_symbol = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"{name} has been renamed to {new_symbol.__name__}. "
            f"Use {new_symbol.__name__} from strands.vended_memory_stores.local instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_symbol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
