"""MemoryManager primitive for cross-session agent memory.

This module provides the :class:`MemoryManager`, a :class:`~strands.plugins.Plugin`
that manages one or more :class:`~strands.memory.types.MemoryStore` backends and
exposes ``search_memory`` / ``add_memory`` tools for agent-driven recall and
persistence, plus programmatic :meth:`MemoryManager.search` / :meth:`MemoryManager.add`
methods. Any tools the stores themselves provide (via
:meth:`MemoryStore.get_tools`) are registered alongside these.

Example:
    ```python
    from strands import Agent
    from strands.memory import MemoryManager

    # Agent gains search_memory (+ add_memory) tools
    agent = Agent(memory_manager=MemoryManager(stores=[my_store], add_tool_config=True))

    # Programmatic access via the instance
    manager = MemoryManager(stores=[my_store], add_tool_config=True)
    agent = Agent(memory_manager=manager)
    await manager.search("user preferences")
    await manager.add("User prefers dark mode")
    ```
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..hooks.events import AfterInvocationEvent
from ..plugins import Plugin, hook
from ..tools.decorator import DecoratedFunctionTool, tool
from .types import (
    MemoryAddToolConfig,
    MemoryEntry,
    MemoryStore,
    MemoryStoreError,
    MemoryToolConfig,
)

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEARCH_RESULTS = 3
"""Default maximum results per store when neither the caller nor the store specifies one."""

_SEARCH_TOOL_DESCRIPTION = (
    "Search long-term memory for facts, preferences, or context from previous conversations. "
    "Use when you need background about the user or topic that may have been discussed before."
)

_ADD_TOOL_DESCRIPTION = (
    "Add facts, preferences, or decisions to long-term memory so they are remembered across "
    "conversations. Use when the user shares something worth recalling later."
)


def _flatten_reasons(reasons: list[Exception]) -> list[Exception]:
    """Flatten nested :class:`MemoryStoreError`s so the leaves are concrete reasons.

    Args:
        reasons: A list of exceptions, some of which may be aggregate errors.

    Returns:
        A flat list with concrete (non-aggregate) exceptions as leaves.
    """
    flat: list[Exception] = []
    for reason in reasons:
        if isinstance(reason, MemoryStoreError) and reason.errors:
            flat.extend(_flatten_reasons(reason.errors))
        else:
            flat.append(reason)
    return flat


def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Convert a memory entry to a JSON-serializable dict, omitting empty fields."""
    result: dict[str, Any] = {"content": entry.content}
    if entry.store_name:
        result["store_name"] = entry.store_name
    if entry.metadata:
        result["metadata"] = entry.metadata
    return result


class MemoryManager(Plugin):
    """Provides cross-session memory retrieval and storage for agents.

    Manages one or more :class:`~strands.memory.types.MemoryStore` backends, exposing
    ``search_memory`` and ``add_memory`` tools for agent-driven recall and persistence.
    Any tools the stores themselves provide (via :meth:`MemoryStore.get_tools`) are
    registered alongside these.

    The manager is a :class:`~strands.plugins.Plugin`: pass it to an agent via the
    dedicated ``memory_manager`` parameter (``Agent(memory_manager=...)``), which
    registers its tools and its write-drain hook.

    Example:
        ```python
        from strands import Agent
        from strands.memory import MemoryManager

        manager = MemoryManager(stores=[my_store], add_tool_config=True)
        agent = Agent(memory_manager=manager)
        await manager.search("user preferences")
        ```
    """

    name = "strands.memory_manager"

    def __init__(
        self,
        stores: list[MemoryStore],
        *,
        search_tool_config: bool | MemoryToolConfig = True,
        add_tool_config: bool | MemoryAddToolConfig = False,
    ) -> None:
        """Initialize the memory manager.

        Args:
            stores: One or more memory stores to manage.
            search_tool_config: Search tool configuration. ``True`` (default) registers
                the ``search_memory`` tool with default name/description; pass a
                :class:`~strands.memory.types.MemoryToolConfig` to customize, or ``False``
                to disable the tool.
            add_tool_config: Add tool configuration. ``False`` (default) disables the
                ``add_memory`` tool. ``True`` lets the tool write to all writable stores;
                pass a :class:`~strands.memory.types.MemoryAddToolConfig` to customize the
                name/description, restrict it to specific stores, or control write awaiting.

        Raises:
            ValueError: If ``stores`` is empty, two stores share a name, a writable store
                does not implement ``add``, the add tool is enabled with no writable stores,
                or ``add_tool_config.stores`` references an unknown or read-only store.
        """
        if not stores:
            raise ValueError("MemoryManager requires at least one store")

        seen_names: set[str] = set()
        for store in stores:
            if store.name in seen_names:
                raise ValueError(f"MemoryManager: duplicate store name '{store.name}'")
            seen_names.add(store.name)

            if store.writable and type(store).add is MemoryStore.add:
                raise ValueError(f"MemoryManager: store '{store.name}' is writable but does not implement add")

        self._stores = list(stores)
        self._search_stores = self._stores
        # All writable stores: the unscoped target set for the programmatic add() method.
        self._add_stores = [s for s in self._stores if s.writable]
        self._pending_writes: set[asyncio.Task[None]] = set()

        self._search_tool_config: MemoryToolConfig | None
        if search_tool_config is False:
            self._search_tool_config = None
        elif isinstance(search_tool_config, MemoryToolConfig):
            self._search_tool_config = search_tool_config
        else:
            self._search_tool_config = MemoryToolConfig()

        self._add_tool_config: MemoryAddToolConfig | None
        if add_tool_config is False:
            self._add_tool_config = None
            self._add_tool_stores: list[MemoryStore] = []
        else:
            if not self._add_stores:
                raise ValueError("MemoryManager: add_tool_config is enabled but no stores are writable")
            self._add_tool_config = (
                add_tool_config if isinstance(add_tool_config, MemoryAddToolConfig) else MemoryAddToolConfig()
            )
            self._add_tool_stores = self._resolve_add_tool_stores(self._add_tool_config)

        # Pre-seed the dynamically named tools so Plugin.__init__ skips auto-discovery
        # (the @tool/@hook scan only finds class-level decorated members; these are built here).
        self._tools: list[DecoratedFunctionTool] = []
        if self._search_tool_config is not None:
            self._tools.append(self._create_search_tool(self._search_tool_config))
        if self._add_tool_config is not None:
            self._tools.append(self._create_add_tool(self._add_tool_config, self._add_tool_stores))

        super().__init__()

    def _resolve_add_tool_stores(self, config: MemoryAddToolConfig) -> list[MemoryStore]:
        """Resolve the writable stores the ``add_memory`` tool may write to.

        When ``config.stores`` is given, each entry (a store name or a
        :class:`MemoryStore` instance) must resolve by name to a configured, writable
        store (else raises). Omitted means all writable stores.

        Args:
            config: The add tool configuration.

        Returns:
            The list of writable stores the add tool is scoped to.

        Raises:
            ValueError: If a named store is not configured or is not writable.
        """
        if config.stores is None:
            return self._add_stores

        names = [s if isinstance(s, str) else s.name for s in config.stores]

        resolved: list[MemoryStore] = []
        for name in dict.fromkeys(names):
            found = self._find_store(name)
            if found is None:
                raise ValueError(f"MemoryManager: add_tool_config store '{name}' not found")
            if not found.writable:
                raise ValueError(f"MemoryManager: add_tool_config store '{name}' is not writable")
            resolved.append(found)
        return resolved

    def _find_store(self, name: str) -> MemoryStore | None:
        """Return the configured store with the given name, or None."""
        return next((s for s in self._stores if s.name == name), None)

    def init_agent(self, agent: Agent) -> None:
        """Register any store-provided tools with the agent.

        The manager's own ``search_memory`` / ``add_memory`` tools are auto-registered
        via the plugin's :attr:`tools`. Store-provided tools are registered here so they
        are not constrained by the plugin's tool list type.

        Args:
            agent: The agent this manager is being attached to.
        """
        for store in self._stores:
            store_tools = store.get_tools()
            if store_tools:
                agent.tool_registry.process_tools(list(store_tools))

    @hook
    async def _drain_pending_writes(self, event: AfterInvocationEvent) -> None:
        """Await fire-and-forget memory writes so they survive event-loop teardown.

        Strands runs invocations through ``run_async``, which can create and close an
        event loop per call. Draining here guarantees dispatched writes complete before
        the invocation returns rather than being cancelled at loop teardown.

        Args:
            event: The after-invocation event (unused).
        """
        if not self._pending_writes:
            return
        await asyncio.gather(*list(self._pending_writes), return_exceptions=True)

    async def search(
        self,
        query: str,
        *,
        max_search_results: int | None = None,
        stores: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """Search stores for entries matching the query.

        This method is unscoped, with full access to all configured stores. When
        ``stores`` is omitted, all stores are searched; when provided, only the named
        stores are searched. Per-store failures are logged and skipped. Each returned
        entry is attributed to its source store via :attr:`MemoryEntry.store_name`.

        Args:
            query: The search query string.
            max_search_results: Maximum results per store. Overrides each store's own
                default; falls back to the store default, then the SDK default of 3.
            stores: Filter to specific stores by name. Omit to search all.

        Returns:
            A list of memory entries from matching stores.

        Raises:
            ValueError: If a named store does not exist.
        """
        logger.debug(
            "query=<%s>, max_search_results=<%s>, stores=<%s> | searching stores",
            query,
            max_search_results,
            stores,
        )

        target_stores = self._resolve_named_stores(stores) if stores is not None else self._stores

        settled = await asyncio.gather(
            *(
                store.search(
                    query,
                    max_search_results=(
                        max_search_results
                        if max_search_results is not None
                        else store.max_search_results
                        if store.max_search_results is not None
                        else DEFAULT_MAX_SEARCH_RESULTS
                    ),
                )
                for store in target_stores
            ),
            return_exceptions=True,
        )

        results: list[MemoryEntry] = []
        for store, outcome in zip(target_stores, settled, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("store=<%s>, reason=<%s> | store search failed", store.name, outcome)
                continue
            for entry in outcome:
                results.append(replace(entry, store_name=store.name))

        logger.debug("results=<%d> | search complete", len(results))
        return results

    async def add(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        stores: list[str] | None = None,
    ) -> None:
        """Add content to writable stores.

        This method is unscoped, with full access to all configured writable stores.
        Target stores are validated first (an unknown or read-only named store raises),
        then the writes are awaited: per-store failures are logged, and a
        :class:`MemoryStoreError` is raised if any store fails.

        Args:
            content: The text content to add.
            metadata: Optional metadata to associate with the entry.
            stores: Filter to specific writable stores by name. Omit to write to all
                writable stores.

        Raises:
            ValueError: If a named store does not exist, is read-only, or no writable
                store matched.
            MemoryStoreError: If one or more store writes fail.
        """
        if stores is not None:
            writable_stores: list[MemoryStore] = []
            for name in dict.fromkeys(stores):
                found = self._find_store(name)
                if found is None:
                    raise ValueError(f"MemoryManager: store '{name}' not found")
                if not found.writable:
                    raise ValueError(f"MemoryManager: store '{name}' is read-only")
                writable_stores.append(found)
        else:
            writable_stores = self._add_stores

        if not writable_stores:
            raise ValueError("MemoryManager: no writable store matched")

        settled = await asyncio.gather(
            *(store.add(content, metadata) for store in writable_stores),
            return_exceptions=True,
        )

        failures: list[Exception] = []
        failed_names: list[str] = []
        for store, outcome in zip(writable_stores, settled, strict=True):
            if isinstance(outcome, Exception):
                logger.warning("store=<%s>, reason=<%s> | store write failed", store.name, outcome)
                failures.append(outcome)
                failed_names.append(store.name)

        if failures:
            raise MemoryStoreError(
                f"MemoryManager: store writes failed: {', '.join(failed_names)}",
                errors=failures,
            )

    def _resolve_named_stores(self, names: list[str]) -> list[MemoryStore]:
        """Resolve store names to configured stores, deduping and preserving order.

        Args:
            names: Store names to resolve.

        Returns:
            The resolved stores.

        Raises:
            ValueError: If a name does not match a configured store.
        """
        resolved: list[MemoryStore] = []
        for name in dict.fromkeys(names):
            found = self._find_store(name)
            if found is None:
                raise ValueError(f"MemoryManager: store '{name}' not found")
            resolved.append(found)
        return resolved

    def _resolve_tool_targets(self, scoped_names: list[str], requested: list[str] | None) -> list[str]:
        """Resolve the store names a tool callback should target against its scoped set.

        - Omitting ``requested`` (or an empty list) targets all scoped stores.
        - In-scope names are kept; out-of-scope names are dropped with a warning.
        - When every requested name is out of scope, raises so the model receives an
          actionable error (the tool layer turns this into a model-visible result).

        Args:
            scoped_names: Store names available to this tool.
            requested: Store names the model asked for, if any.

        Returns:
            A non-empty list of in-scope store names to target.

        Raises:
            ValueError: If every requested name is out of scope.
        """
        if not requested:
            return scoped_names

        in_scope = [name for name in requested if name in scoped_names]
        out_of_scope = [name for name in requested if name not in scoped_names]

        if not in_scope:
            raise ValueError(
                f"MemoryManager: requested=<{', '.join(requested)}> | "
                f"none of the requested memory stores are available; available stores: {', '.join(scoped_names)}"
            )

        if out_of_scope:
            logger.warning(
                "requested=<%s> | ignoring memory stores outside this tool's scope",
                ", ".join(out_of_scope),
            )

        return in_scope

    def _create_search_tool(self, config: MemoryToolConfig) -> DecoratedFunctionTool:
        """Build the ``search_memory`` tool, scoped to all searchable stores."""
        description = config.description or _SEARCH_TOOL_DESCRIPTION
        store_descriptions = [f"- {s.name}: {s.description}" for s in self._search_stores if s.description]
        if store_descriptions:
            description += "\n\nAvailable memory stores:\n" + "\n".join(store_descriptions)
            description += (
                "\n\nYou can target one or more memory stores by name if you know which domains are "
                "relevant, or omit the stores parameter to search all."
            )

        scoped_names = [s.name for s in self._search_stores]
        manager = self

        async def search_memory(
            query: str,
            max_search_results: int | None = None,
            stores: list[str] | None = None,
        ) -> dict[str, Any]:
            """Search long-term memory for relevant entries.

            Args:
                query: What to search for.
                max_search_results: Maximum number of results per store.
                stores: Filter to specific stores by name. Omit to search all available stores.
            """
            targets = manager._resolve_tool_targets(scoped_names, stores)
            results = await manager.search(query, max_search_results=max_search_results, stores=targets)
            return {"results": [_entry_to_dict(entry) for entry in results]}

        return tool(name=config.name or "search_memory", description=description)(search_memory)

    def _create_add_tool(self, config: MemoryAddToolConfig, stores: list[MemoryStore]) -> DecoratedFunctionTool:
        """Build the ``add_memory`` tool, scoped to its writable store allowlist."""
        description = config.description or _ADD_TOOL_DESCRIPTION
        store_descriptions = [f"- {s.name}: {s.description}" for s in stores if s.description]
        if store_descriptions:
            description += "\n\nAvailable writable stores:\n" + "\n".join(store_descriptions)
            description += (
                "\n\nYou can target a specific store by name to route facts to the right place, "
                "or omit to add to all available writable stores."
            )

        scoped_names = [s.name for s in stores]
        wait_for_writes = config.wait_for_writes
        manager = self

        async def add_memory(
            entries: list[str],
            stores: list[str] | None = None,
        ) -> dict[str, Any]:
            """Add one or more entries to long-term memory.

            Args:
                entries: Data to add to long-term memory.
                stores: Target specific stores by name. Omit to add to all writable stores.
            """
            if not entries:
                raise ValueError("MemoryManager: entries must not be empty")

            targets = manager._resolve_tool_targets(scoped_names, stores)

            if not wait_for_writes:
                # Fire-and-forget: dispatch without awaiting so the agent loop is not blocked.
                # Failures are logged; pending writes are drained at the end of the invocation.
                for content in entries:
                    manager._dispatch_write(content, targets)
                return {"accepted": len(entries)}

            # Await mode: surface failures to the model with concrete reasons.
            settled = await asyncio.gather(
                *(manager.add(content, stores=targets) for content in entries),
                return_exceptions=True,
            )
            failures = [outcome for outcome in settled if isinstance(outcome, Exception)]
            if failures:
                reasons = _flatten_reasons(failures)
                raise MemoryStoreError(
                    f"MemoryManager: failed to add {len(failures)} of {len(entries)} entries: "
                    + "; ".join(str(reason) for reason in reasons),
                    errors=reasons,
                )

            return {"stored": len(entries)}

        return tool(name=config.name or "add_memory", description=description)(add_memory)

    def _dispatch_write(self, content: str, stores: list[str]) -> None:
        """Dispatch a fire-and-forget write, holding a strong task reference.

        The event loop keeps only a weak reference to tasks, so the reference is held in
        :attr:`_pending_writes` (and discarded on completion) to prevent mid-flight GC.

        Args:
            content: The text content to add.
            stores: The target store names.
        """
        task = asyncio.create_task(self._safe_write(content, stores))
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

    async def _safe_write(self, content: str, stores: list[str]) -> None:
        """Run a write, swallowing and logging any failure (fire-and-forget mode)."""
        try:
            await self.add(content, stores=stores)
        except Exception:
            logger.warning("stores=<%s> | fire-and-forget memory write failed", ", ".join(stores), exc_info=True)
