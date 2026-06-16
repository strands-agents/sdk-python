"""Vended :class:`~strands.memory.types.MemoryStore` backends for Strands Agents.

Each store here is an optional, self-contained implementation of the
:class:`~strands.memory.types.MemoryStore` Protocol that a
:class:`~strands.memory.memory_manager.MemoryManager` can manage. Stores that
depend on a third-party backend pull it from an ``extras`` group and import it
lazily, so importing this subpackage never requires the optional dependency to
be installed.

Example:
    ```python
    from strands.memory.stores import GraphitiMemoryStore
    ```
"""

from .graphiti import GraphitiMemoryStore

__all__ = ["GraphitiMemoryStore"]
