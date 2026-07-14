"""Unified storage module.

Provides the :class:`Storage` protocol and shipped implementations for persisting
raw bytes under string keys. It is the SDK's single low-level persistence
primitive — a minimal four-operation contract (``write``, ``read``, ``delete``,
``list``) over opaque ``bytes`` values, keyed by ``/``-separated path-like strings.

Example:
    ```python
    from strands.storage import InMemoryStorage, LocalFileStorage, S3Storage
    ```
"""

from .base import Storage
from .in_memory_storage import InMemoryStorage
from .local_file_storage import LocalFileStorage
from .s3_storage import S3Storage

__all__ = [
    "Storage",
    "InMemoryStorage",
    "LocalFileStorage",
    "S3Storage",
]
