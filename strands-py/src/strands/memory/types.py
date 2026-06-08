"""Type definitions and interfaces for the memory module.

This module defines the data types and the ``MemoryStore`` interface that the
:class:`~strands.memory.memory_manager.MemoryManager` builds on. A memory store
is a searchable (and optionally writable) backend for an agent's long-term
memory; concrete stores subclass :class:`MemoryStore`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types.tools import AgentTool


@dataclass
class MemoryEntry:
    """A single memory entry retrieved from or stored to a memory store.

    Attributes:
        content: The textual content of this memory entry.
        store_name: Name of the store this entry came from. Populated by
            :meth:`MemoryManager.search` so callers (and the model, via
            ``search_memory``) can tell which store produced each result and refine
            targeting. Stores need not set this themselves.
        metadata: Optional metadata (e.g., score, source, id, timestamp).
    """

    content: str
    store_name: str | None = None
    metadata: dict[str, Any] | None = None


class MemoryStore(ABC):
    """Interface for a memory store backend.

    Every store is searchable; the ``writable`` flag declares whether it also
    accepts writes, which is how the :class:`~strands.memory.memory_manager.MemoryManager`
    decides where to route them. ``search_memory`` can query all stores, while
    ``add_memory`` can only write to ``writable`` stores.

    Concrete stores subclass this and implement :meth:`search`. Writable stores
    (``writable=True``) must additionally override :meth:`add`.

    Example:
        ```python
        from strands.memory import MemoryEntry, MemoryStore


        class MyStore(MemoryStore):
            def __init__(self) -> None:
                super().__init__(name="my-store", writable=True)
                self._entries: list[str] = []

            async def search(self, query, *, max_search_results=None):
                return [MemoryEntry(content=e) for e in self._entries if query in e]

            async def add(self, content, metadata=None):
                self._entries.append(content)
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        writable: bool = False,
        description: str | None = None,
        max_search_results: int | None = None,
    ) -> None:
        """Initialize the memory store.

        Args:
            name: Identifier for this store, used to target specific stores in the
                search/add tools. Must be unique within a ``MemoryManager``.
            writable: Whether this store accepts writes. When True, the subclass
                must override :meth:`add`. Defaults to False.
            description: Human-readable description of what this store contains.
                Included in tool descriptions when present.
            max_search_results: Default maximum number of results this store returns
                per search, used when a caller does not pass a per-call
                ``max_search_results``.
        """
        self.name = name
        self.writable = writable
        self.description = description
        self.max_search_results = max_search_results

    @abstractmethod
    async def search(self, query: str, *, max_search_results: int | None = None) -> list[MemoryEntry]:
        """Search the store for entries matching the query, ordered by relevance.

        Args:
            query: The search query string.
            max_search_results: Maximum number of results to return from this store.

        Returns:
            A list of matching memory entries.
        """
        ...

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add content to the store.

        Must be overridden by stores that declare ``writable=True``. The default
        implementation raises, so a writable store that forgets to implement it
        fails loudly.

        Args:
            content: The text content to add.
            metadata: Optional metadata to associate with the entry.

        Raises:
            NotImplementedError: If the store does not override this method.
        """
        raise NotImplementedError(f"store '{self.name}' does not implement add")

    def get_tools(self) -> list[AgentTool]:
        """Return store-specific tools to register with the agent.

        Registered alongside the manager's ``search_memory`` / ``add_memory`` tools.
        Override to expose backend-specific capabilities (e.g. a store-native query
        tool). Optional; mirrors :meth:`Plugin.tools`.

        Returns:
            A list of tools provided by this store. Empty by default.
        """
        return []


@dataclass
class MemoryToolConfig:
    """Configuration for customizing a memory tool's name or description.

    Attributes:
        name: Custom tool name. Defaults to the tool's standard name when None.
        description: Custom tool description. Defaults to the tool's standard
            description when None.
    """

    name: str | None = None
    description: str | None = None


@dataclass
class MemoryAddToolConfig(MemoryToolConfig):
    """Configuration for the ``add_memory`` tool.

    Extends :class:`MemoryToolConfig` with an explicit allowlist of stores the tool
    may write to and a flag controlling whether writes are awaited.

    Attributes:
        stores: The writable stores the ``add_memory`` tool may write to, given as
            store names or :class:`MemoryStore` instances. Each must be a configured,
            writable store. Omit to allow all writable stores.
        wait_for_writes: Whether the tool waits for store writes before returning to
            the model. Defaults to True.

            - True (default): waits for writes. The tool returns ``{"stored": N}`` on
              success, or surfaces a failure to the model if any store write fails.
            - False: fire-and-forget. The tool returns ``{"accepted": N}`` once writes
              are dispatched (so a slow backend never blocks the agent loop); per-store
              failures are logged. Dispatched writes are drained at the end of the agent
              invocation so they are not lost to event-loop teardown.
    """

    stores: list[str | MemoryStore] | None = field(default=None)
    wait_for_writes: bool = True


class MemoryStoreError(Exception):
    """Raised when one or more memory store operations fail.

    Aggregates the underlying per-store errors (the Python analogue of an
    ``AggregateError``), since the standard library ``ExceptionGroup`` is only
    available on Python 3.11+.

    Attributes:
        errors: The underlying exceptions raised by individual stores.
    """

    def __init__(self, message: str, errors: list[Exception] | None = None) -> None:
        """Initialize the error.

        Args:
            message: A human-readable summary, typically naming the failed stores.
            errors: The underlying per-store exceptions, if any.
        """
        super().__init__(message)
        self.errors = errors if errors is not None else []
