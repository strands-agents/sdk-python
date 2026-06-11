"""Core types for the Strands memory module.

This module defines the data shapes and the store contract that the rest of the
memory subsystem builds on: the :class:`MemoryEntry` record, the option/config
dataclasses used by the manager and its tools, and the :class:`MemoryStore`
runtime contract that pluggable backends implement.

These are Python ports of the TypeScript ``memory/types.ts`` interfaces. All
public field and method names use ``snake_case`` per the requirements naming
convention mapping (e.g. ``maxSearchResults`` -> ``max_search_results``,
``addMessages`` -> ``add_messages``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..types.content import Message
from ..types.tools import AgentTool

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import with the extraction subpackage,
    # whose ``types`` module imports from this one. ``ExtractionConfig`` is only
    # referenced in annotations, so a ``TYPE_CHECKING`` import is sufficient.
    from .extraction.types import ExtractionConfig

# JSON-compatible metadata mapping. Modeled as ``dict[str, Any]`` because memory
# metadata holds arbitrary JSON values (scores, ids, timestamps, etc.).
Metadata = dict[str, Any]


@dataclass
class MemoryEntry:
    """A single memory entry retrieved from or stored to a memory store.

    Attributes:
        content: The textual content of this memory entry.
        store_name: Name of the store this entry came from. Populated by
            ``MemoryManager.search`` so callers (and the model, via the search
            tool) can tell which store produced each result and refine
            targeting. Stores need not set this themselves.
        metadata: Optional metadata (e.g., score, source, id, timestamp).
    """

    content: str
    store_name: str | None = None
    metadata: Metadata | None = None


@dataclass
class SearchOptions:
    """Options passed to :meth:`MemoryStore.search`.

    Store implementations may extend this with backend-specific fields in their
    own ``search`` signature. Note that ``MemoryManager.search`` only forwards
    the base fields here across its (potentially heterogeneous) stores -- to use
    a store's extended options, call that store's ``search`` directly, or set
    them as per-instance defaults on the store.

    Attributes:
        max_search_results: Maximum number of results to return from this store.
    """

    max_search_results: int | None = None


@dataclass
class AddMessagesContext:
    """Context the manager supplies to :meth:`MemoryStore.add_messages`.

    This is an extension point: it is intentionally empty for now so fields can
    be added later without a breaking signature change.
    """


@dataclass
class MemorySearchOptions(SearchOptions):
    """Options for ``MemoryManager.search``.

    Extends the store primitive :class:`SearchOptions` with manager-level store
    routing.

    Attributes:
        stores: Filter to specific stores by name. Omit to search all. Note: a
            programmatic ``MemoryManager.search`` with an empty list searches no
            stores (returns ``[]``), whereas the ``search_memory`` tool treats an
            empty list as "search all in-scope stores".
    """

    stores: list[str] | None = None


@dataclass
class MemoryAddOptions:
    """Options for ``MemoryManager.add``.

    Attributes:
        metadata: Metadata to associate with the added entry.
        stores: Filter to specific writable stores by name. Omit to write to all
            writable stores. Note: a programmatic ``MemoryManager.add`` with an
            empty list matches no store (raises), whereas the ``add_memory`` tool
            treats an empty list as "write to all in-scope stores".
    """

    metadata: Metadata | None = None
    stores: list[str] | None = None


@dataclass
class MemoryToolConfig:
    """Configuration for customizing a memory tool's name or description.

    Attributes:
        name: Custom tool name.
        description: Custom tool description.
    """

    name: str | None = None
    description: str | None = None


@dataclass
class MemoryAddToolConfig(MemoryToolConfig):
    """Configuration for the ``add_memory`` tool.

    Extends :class:`MemoryToolConfig` with an explicit allowlist of stores the
    tool may write to.

    Attributes:
        stores: The writable stores the ``add_memory`` tool may write to, given
            as store names or :class:`MemoryStore` instances. Each must be a
            configured, ``writable`` store. Omit (or set ``add_tool_config`` to
            ``True``) to allow all writable stores.
        wait_for_writes: Whether the tool waits for store writes before returning
            to the model. Defaults to ``True``.

            - ``True`` (default): waits for writes -- the tool returns
              ``{"stored": ...}`` on success, or surfaces a failure to the model
              if any store write fails.
            - ``False``: fire-and-forget -- the tool returns ``{"accepted": ...}``
              once writes are dispatched (so a slow backend never blocks the
              agent loop); per-store failures are logged.
    """

    stores: list[str | MemoryStore] | None = None
    wait_for_writes: bool = True


@dataclass
class MemoryManagerConfig:
    """Configuration for the ``MemoryManager``.

    Provided as a documentation/typing aid that mirrors the constructor kwargs;
    the manager itself accepts these fields directly as keyword arguments.

    Attributes:
        stores: One or more memory stores to manage.
        search_tool_config: Search tool configuration. Defaults to ``True``.
        add_tool_config: Add tool configuration. Defaults to ``False`` (opt-in).
            ``True`` lets the tool write to all writable stores; pass a
            :class:`MemoryAddToolConfig` with ``stores`` to restrict it to
            specific ones.
        flush_on_invocation_end: When ``True``, await pending extraction writes
            at the end of each agent invocation. Defaults to ``False``.
    """

    stores: list[MemoryStore]
    search_tool_config: MemoryToolConfig | bool = True
    add_tool_config: MemoryAddToolConfig | bool = False
    flush_on_invocation_end: bool = False


@dataclass
class MemoryStoreConfig:
    """Declarative config shape shared by every memory store.

    This captures a store's identity and behavior knobs. It is the config-shape
    counterpart to the runtime :class:`MemoryStore` contract: the runtime
    protocol declares the same fields alongside its methods, while this type
    documents the plain configuration shape for export parity with the
    TypeScript module. Concrete stores add their own backend-specific config
    fields on top.

    Attributes:
        name: Identifier for this store, used to target specific stores in
            search/add tools. Must be unique.
        description: Human-readable description of what this store contains.
            Included in tool descriptions.
        max_search_results: Default maximum number of results this store returns
            per search, used when a caller does not pass a per-call
            ``max_search_results``.
        writable: Whether this store accepts writes. Semantically defaults to
            ``False`` (caller intent); concrete stores resolve it to a definite
            boolean on the :class:`MemoryStore` contract.
        extraction: Automatic-extraction configuration for this store. When set,
            the manager runs the configured triggers and writes extracted (or,
            with no extractor, raw) messages to this store. Requires the store to
            be writable. Omit for a purely tool-driven store.
    """

    name: str
    description: str | None = None
    max_search_results: int | None = None
    writable: bool = False
    extraction: ExtractionConfig | None = None


@runtime_checkable
class MemoryStore(Protocol):
    """Runtime contract for a memory store backend.

    Every store is searchable; the resolved ``writable`` flag declares whether it
    also accepts writes, which is how the ``MemoryManager`` decides where to route
    them. The search tool can query all stores, while the add tool can only write
    to ``writable`` stores.

    A store author implements the identity attributes plus :meth:`search`, and
    optionally one or more of :meth:`add`, :meth:`add_messages`, and
    :meth:`get_tools`.

    Attributes:
        name: Unique identifier for this store, used to target it in
            search/add tools.
        description: Human-readable description of what this store contains.
        max_search_results: Default maximum number of results returned per
            search.
        writable: Whether this store accepts writes. Defaults to ``False``
            semantically (searchable only). A store declaring ``writable=True``
            requires at least one write sink -- :meth:`add`, :meth:`add_messages`,
            or both -- to be implemented.
        extraction: Optional automatic-extraction configuration. Requires the
            store to be writable.
    """

    name: str
    description: str | None
    max_search_results: int | None
    writable: bool
    extraction: ExtractionConfig | None

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Search the store for entries matching the query, ordered by relevance.

        Args:
            query: The search query.
            options: Optional search options (e.g. ``max_search_results``).

        Returns:
            Matching entries, ordered by relevance.
        """
        ...

    # --- Optional methods -------------------------------------------------
    # The following are optional. A store implements them only when it supports
    # the corresponding capability; use ``_has_method`` / ``_has_write_sink`` to
    # detect presence rather than relying on the Protocol defaults below.

    async def add(self, content: str, metadata: Metadata | None = None) -> Any:
        """Add a single piece of content to the store.

        Used by the add tool, programmatic ``MemoryManager.add``, and by
        extraction when an extractor produces discrete entries. Extraction writes
        are at-least-once, so implementations used with extraction should tolerate
        duplicate writes.

        The resolved value is store-specific; the manager does not consume it.

        Args:
            content: The content to store.
            metadata: Optional metadata to associate with the entry.

        Returns:
            A store-specific value (e.g. a created record id or write receipt).
        """
        ...

    async def add_messages(self, messages: list[Message], context: AddMessagesContext | None = None) -> Any:
        """Ingest a batch of conversation messages, preserving role structure.

        This is the sink for automatic extraction that does not distill facts
        client-side: the manager hands the filtered message batch straight here
        in one call. The resolved value is store-specific and not consumed by the
        manager.

        Args:
            messages: The filtered messages to ingest, in order.
            context: Manager-supplied per-batch context.

        Returns:
            A store-specific value.
        """
        ...

    def get_tools(self) -> list[AgentTool]:
        """Return store-specific tools to register with the agent.

        Implement to expose backend-specific capabilities (e.g. a store-native
        query tool). Registered alongside the manager's search/add tools.

        Returns:
            Tools provided by this store.
        """
        ...


def _has_method(store: object, name: str) -> bool:
    """Return whether ``store`` actually implements the named method.

    Optional store methods (``add``, ``add_messages``, ``get_tools``) are
    detected by inspecting the store's own type rather than the instance, so a
    method is only considered present when the concrete store class implements
    it -- not when it is merely the :class:`MemoryStore` Protocol's default stub.

    Args:
        store: The store instance to inspect.
        name: The method name to look for.

    Returns:
        ``True`` if the store's type implements ``name`` with a callable that is
        distinct from the Protocol default; otherwise ``False``.
    """
    method = getattr(type(store), name, None)
    if method is None:
        return False
    # A class that explicitly subclasses the ``MemoryStore`` Protocol can inherit
    # the Protocol's stub for an optional method; treat that as "not implemented".
    if method is getattr(MemoryStore, name, None):
        return False
    return callable(method)


def _has_write_sink(store: MemoryStore) -> bool:
    """Return whether ``store`` provides at least one write sink.

    A writable store must implement ``add``, ``add_messages``, or both.

    Args:
        store: The store to check.

    Returns:
        ``True`` if the store implements ``add`` or ``add_messages``.
    """
    return _has_method(store, "add") or _has_method(store, "add_messages")
