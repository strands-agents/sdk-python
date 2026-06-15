"""Cross-session memory retrieval and storage for agents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..hooks.events import MessageAddedEvent
from ..plugins.plugin import Plugin
from ..tools.decorator import tool
from ..types.exceptions import AggregateMemoryError
from ..types.tools import AgentTool
from .extraction.coordinator import ExtractionCoordinator
from .extraction.types import ExtractionTrigger, ExtractionTriggerContext
from .types import (
    MemoryAddOptions,
    MemoryAddToolConfig,
    MemoryEntry,
    MemorySearchOptions,
    MemoryStore,
    MemoryToolConfig,
    _has_method,
    _has_write_sink,
)

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)

SEARCH_TOOL_DESCRIPTION = (
    "Search long-term memory for facts, preferences, or context from previous conversations. Use when you need "
    "background about the user or topic that may have been discussed before."
)

ADD_TOOL_DESCRIPTION = (
    "Add facts, preferences, or decisions to long-term memory so they are remembered across conversations. Use when "
    "the user shares something worth recalling later."
)

# Default maximum results per store when neither caller nor store specifies one.
DEFAULT_MAX_SEARCH_RESULTS = 3


def _normalize_triggers(trigger: ExtractionTrigger | list[ExtractionTrigger]) -> list[ExtractionTrigger]:
    """Normalize a store's ``trigger`` field (a single trigger or a list) to a list."""
    return list(trigger) if isinstance(trigger, list) else [trigger]


def _flatten_reasons(reasons: list[BaseException]) -> list[BaseException]:
    """Flatten nested aggregate errors so the leaves are concrete reasons."""
    flattened: list[BaseException] = []
    for reason in reasons:
        if isinstance(reason, AggregateMemoryError):
            flattened.extend(_flatten_reasons(reason.errors))
        else:
            flattened.append(reason)
    return flattened


class MemoryManager(Plugin):
    """Provides cross-session memory retrieval and storage for agents.

    Example:
        ```python
        from strands import Agent
        from strands.memory import MemoryManager

        memory_manager = MemoryManager(stores=[my_store])
        agent = Agent(model=model, memory_manager=memory_manager)
        agent("Remember I prefer dark mode")

        results = await memory_manager.search("user preferences")
        ```
    """

    name = "strands:memory-manager"

    def __init__(
        self,
        stores: list[MemoryStore],
        search_tool_config: MemoryToolConfig | bool = True,
        add_tool_config: MemoryAddToolConfig | bool = False,
    ) -> None:
        """Initialize the memory manager.

        Args:
            stores: One or more memory stores to manage.
            search_tool_config: Search tool configuration. ``True`` (default)
                registers a ``search_memory`` tool with default name/description;
                a :class:`MemoryToolConfig` customizes it; ``False`` disables it.
            add_tool_config: Add tool configuration. ``False`` (default) disables
                the add tool; ``True`` lets it write to all writable stores; a
                :class:`MemoryAddToolConfig` restricts/customizes it.

        Raises:
            ValueError: If ``stores`` is empty, a store name is duplicated, a
                writable store has no write sink, an extraction config is
                misconfigured, or the add tool is enabled/scoped against stores
                that cannot accept discrete ``add`` writes.
        """
        if len(stores) == 0:
            raise ValueError("MemoryManager: at least one store is required")

        seen_names: set[str] = set()
        for store in stores:
            if store.name in seen_names:
                raise ValueError(f"MemoryManager: duplicate store name '{store.name}'")
            seen_names.add(store.name)

            if store.writable and not _has_write_sink(store):
                raise ValueError(
                    f"MemoryManager: store '{store.name}' is writable but has no add or add_messages method"
                )

            if store.extraction is not None:
                if not store.writable:
                    raise ValueError(f"MemoryManager: store '{store.name}' has extraction config but is not writable")
                if len(_normalize_triggers(store.extraction.trigger)) == 0:
                    raise ValueError(f"MemoryManager: store '{store.name}' has extraction config but no triggers")
                # Each extraction shape needs its matching write sink.
                if store.extraction.extractor is not None:
                    if not _has_method(store, "add"):
                        raise ValueError(
                            f"MemoryManager: store '{store.name}' has an extractor but no add method "
                            "(extracted entries are written via add)"
                        )
                elif not _has_method(store, "add_messages"):
                    raise ValueError(
                        f"MemoryManager: store '{store.name}' has extraction config without an extractor "
                        "but no add_messages method"
                    )

        super().__init__()

        self._stores = list(stores)
        self._search_stores = list(stores)
        # `add`-targeting paths (tool / programmatic) need an `add` method specifically.
        self._add_stores = [store for store in stores if store.writable and _has_method(store, "add")]
        self._extraction_stores = [store for store in stores if store.writable and store.extraction is not None]

        self._search_tool_config: MemoryToolConfig | bool
        if search_tool_config is False:
            self._search_tool_config = False
        elif isinstance(search_tool_config, MemoryToolConfig):
            self._search_tool_config = search_tool_config
        else:
            self._search_tool_config = MemoryToolConfig()

        self._add_tool_config: MemoryAddToolConfig | bool
        self._add_tool_stores: list[MemoryStore]
        if add_tool_config is None or add_tool_config is False:
            self._add_tool_config = False
            self._add_tool_stores = []
        else:
            # The `add_memory` tool writes via `add`, so needs an `add`-capable store.
            if len(self._add_stores) == 0:
                raise ValueError("MemoryManager: add_tool_config is enabled but no writable stores implement add")
            resolved_config = (
                add_tool_config if isinstance(add_tool_config, MemoryAddToolConfig) else MemoryAddToolConfig()
            )
            self._add_tool_config = resolved_config
            self._add_tool_stores = self._resolve_add_tool_stores(resolved_config)

        # Fire-and-forget background tasks, retained so they aren't GC'd mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

        # Extraction coordinator, created in ``init_agent`` when configured.
        self._coordinator: ExtractionCoordinator | None = None

        # Build tools now; surfaced via the ``tools`` property.
        self._memory_tools: list[AgentTool] = self._build_tools()

    def _resolve_add_tool_stores(self, tool_config: MemoryAddToolConfig) -> list[MemoryStore]:
        """Resolve the writable stores the ``add_memory`` tool may write to.

        Each entry (a store name or instance) must resolve by name to a
        configured, ``add``-capable writable store. Omitted means all such stores.

        Raises:
            ValueError: If a referenced store is not configured, not writable, or
                has no ``add`` method.
        """
        if tool_config.stores is None:
            return self._add_stores

        names = [store if isinstance(store, str) else store.name for store in tool_config.stores]

        resolved: list[MemoryStore] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            found = next((store for store in self._stores if store.name == name), None)
            if found is None:
                raise ValueError(f"MemoryManager: add_tool_config store '{name}' not found")
            if not found.writable:
                raise ValueError(f"MemoryManager: add_tool_config store '{name}' is not writable")
            if not _has_method(found, "add"):
                raise ValueError(f"MemoryManager: add_tool_config store '{name}' has no add method (only add_messages)")
            resolved.append(found)
        return resolved

    def _build_tools(self) -> list[AgentTool]:
        """Build the tools this plugin registers.

        Includes the manager's ``search_memory`` / ``add_memory`` tools plus any
        tools the stores expose via
        :meth:`~strands.memory.types.MemoryStore.get_tools`, in store order.
        """
        tools: list[AgentTool] = []

        if isinstance(self._search_tool_config, MemoryToolConfig):
            tools.append(self._create_search_tool(self._search_tool_config))

        if isinstance(self._add_tool_config, MemoryAddToolConfig):
            tools.append(self._create_add_tool(self._add_tool_config, self._add_tool_stores))

        for store in self._stores:
            if _has_method(store, "get_tools"):
                tools.extend(store.get_tools())

        return tools

    @property
    def tools(self) -> list[AgentTool]:  # type: ignore[override]
        """Tools registered by this plugin: search/add plus any store-provided tools.

        Widens the base :class:`~strands.plugins.plugin.Plugin` annotation because
        a store's ``get_tools`` may contribute any
        :class:`~strands.types.tools.AgentTool`.
        """
        return list(self._memory_tools)

    async def search(self, query: str, options: MemorySearchOptions | None = None) -> list[MemoryEntry]:
        """Search stores for entries matching the query.

        Unscoped: searches all configured stores when ``options.stores`` is
        omitted. Results are attributed to their store via ``store_name`` and
        concatenated in target order.

        Raises:
            ValueError: If a named store is not found (raised before querying).
        """
        requested_stores = options.stores if options is not None else None
        caller_max = options.max_search_results if options is not None else None

        logger.debug(
            "query=<%s>, max_search_results=<%s>, stores=<%s> | searching stores",
            query,
            caller_max,
            requested_stores,
        )

        if requested_stores is not None:
            target_stores: list[MemoryStore] = []
            seen: set[str] = set()
            for name in requested_stores:
                if name in seen:
                    continue
                seen.add(name)
                found = next((store for store in self._stores if store.name == name), None)
                if found is None:
                    raise ValueError(f"MemoryManager: store '{name}' not found")
                target_stores.append(found)
        else:
            target_stores = self._stores

        settled = await asyncio.gather(
            *(
                store.search(
                    query,
                    MemorySearchOptions(
                        max_search_results=(
                            caller_max
                            if caller_max is not None
                            else store.max_search_results
                            if store.max_search_results is not None
                            else DEFAULT_MAX_SEARCH_RESULTS
                        )
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
                results.append(MemoryEntry(content=entry.content, store_name=store.name, metadata=entry.metadata))

        logger.debug("results=<%s> | search complete", len(results))
        return results

    async def add(self, content: str, options: MemoryAddOptions | None = None) -> None:
        """Add content to writable stores.

        Unscoped: targets all configured writable stores. Target stores are
        validated first, then writes are awaited concurrently; per-store failures
        are logged and surfaced as an
        :class:`~strands.types.exceptions.AggregateMemoryError`.

        Raises:
            ValueError: If a named store is not found or is read-only, or if no
                writable store matched.
            AggregateMemoryError: If any targeted store write fails.
        """
        requested_stores = options.stores if options is not None else None
        metadata = options.metadata if options is not None else None

        if requested_stores is not None:
            writable_stores: list[MemoryStore] = []
            seen: set[str] = set()
            for name in requested_stores:
                if name in seen:
                    continue
                seen.add(name)
                found = next((store for store in self._stores if store.name == name), None)
                if found is None:
                    raise ValueError(f"MemoryManager: store '{name}' not found")
                if not found.writable:
                    raise ValueError(f"MemoryManager: store '{name}' is read-only")
                writable_stores.append(found)
        else:
            writable_stores = self._add_stores

        if len(writable_stores) == 0:
            raise ValueError("MemoryManager: no writable store matched")

        settled = await asyncio.gather(
            *(store.add(content, metadata) for store in writable_stores),
            return_exceptions=True,
        )

        failed_names: list[str] = []
        reasons: list[BaseException] = []
        for store, outcome in zip(writable_stores, settled, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("store=<%s>, reason=<%s> | store write failed", store.name, outcome)
                failed_names.append(store.name)
                reasons.append(outcome)

        if failed_names:
            raise AggregateMemoryError(
                f"MemoryManager: store writes failed: {', '.join(failed_names)}",
                reasons,
            )

    def _resolve_tool_targets(self, scoped_names: list[str], requested: list[str] | None) -> list[str]:
        """Resolve the store names a tool callback should target.

        Omitting ``requested`` targets all scoped stores; in-scope names are kept
        and out-of-scope names are dropped with a warning.

        Raises:
            ValueError: If every requested name is out of scope.
        """
        if requested is None or len(requested) == 0:
            return scoped_names

        scoped_set = set(scoped_names)
        in_scope = [name for name in requested if name in scoped_set]
        out_of_scope = [name for name in requested if name not in scoped_set]

        if len(in_scope) == 0:
            raise ValueError(
                f"MemoryManager: requested=<{', '.join(requested)}> | none of the requested memory stores "
                f"are available; available stores: {', '.join(scoped_names)}"
            )

        if out_of_scope:
            logger.warning(
                "requested=<%s> | ignoring memory stores outside this tool's scope",
                ", ".join(out_of_scope),
            )

        return in_scope

    def _create_search_tool(self, config: MemoryToolConfig) -> AgentTool:
        """Build the ``search_memory`` tool."""
        description = config.description if config.description is not None else SEARCH_TOOL_DESCRIPTION
        store_descriptions = [
            f"- {store.name}: {store.description}" for store in self._search_stores if store.description
        ]
        if store_descriptions:
            description += "\n\nAvailable memory stores:\n" + "\n".join(store_descriptions)
            description += (
                "\n\nYou can target one or more memory stores by name if you know which domains are relevant, "
                "or omit the stores parameter to search all."
            )

        scoped_names = [store.name for store in self._search_stores]

        async def search_memory(
            query: str,
            max_search_results: int | None = None,
            stores: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            """Search long-term memory.

            Args:
                query: What to search for.
                max_search_results: Maximum number of results per store.
                stores: Filter to specific stores by name. Omit to search all
                    available stores.

            Returns:
                Matching memory entries, each attributed to its store.
            """
            targets = self._resolve_tool_targets(scoped_names, stores)
            results = await self.search(
                query,
                MemorySearchOptions(max_search_results=max_search_results, stores=targets),
            )
            payload: list[dict[str, Any]] = []
            for entry in results:
                item: dict[str, Any] = {"content": entry.content}
                if entry.store_name:
                    item["store_name"] = entry.store_name
                if entry.metadata:
                    item["metadata"] = entry.metadata
                payload.append(item)
            return payload

        return tool(
            name=config.name if config.name is not None else "search_memory",
            description=description,
        )(search_memory)

    def _create_add_tool(self, config: MemoryAddToolConfig, stores: list[MemoryStore]) -> AgentTool:
        """Build the ``add_memory`` tool."""
        description = config.description if config.description is not None else ADD_TOOL_DESCRIPTION
        store_descriptions = [f"- {store.name}: {store.description}" for store in stores if store.description]
        if store_descriptions:
            description += "\n\nAvailable writable stores:\n" + "\n".join(store_descriptions)
            description += (
                "\n\nYou can target a specific store by name to route facts to the right place, "
                "or omit to add to all available writable stores."
            )

        scoped_names = [store.name for store in stores]
        wait_for_writes = config.wait_for_writes

        async def add_memory(entries: list[str], stores: list[str] | None = None) -> dict[str, int]:
            """Add data to long-term memory.

            Args:
                entries: Data to add to long-term memory.
                stores: Target specific stores by name. Omit to add to all
                    writable stores.

            Returns:
                A summary of the write (``{"stored": n}`` or ``{"accepted": n}``).
            """
            # @tool validation does not enforce ``minItems``, so guard here.
            if not entries:
                raise ValueError("MemoryManager: add_memory requires at least one entry")

            targets = self._resolve_tool_targets(scoped_names, stores)

            if not wait_for_writes:
                # Fire-and-forget: dispatch without awaiting. ``add`` logs per-store failures.
                for content in entries:
                    self._schedule_background(self._add_swallow(content, targets))
                return {"accepted": len(entries)}

            # Await mode: surface failures with concrete (flattened) reasons.
            settled = await asyncio.gather(
                *(self.add(content, MemoryAddOptions(stores=targets)) for content in entries),
                return_exceptions=True,
            )
            failures = [outcome for outcome in settled if isinstance(outcome, BaseException)]
            if failures:
                flattened = _flatten_reasons(failures)
                joined = "; ".join(str(reason) for reason in flattened)
                raise AggregateMemoryError(
                    f"MemoryManager: failed to add {len(failures)} of {len(entries)} entries: {joined}",
                    flattened,
                )

            return {"stored": len(entries)}

        return tool(
            name=config.name if config.name is not None else "add_memory",
            description=description,
        )(add_memory)

    async def _add_swallow(self, content: str, targets: list[str]) -> None:
        """Run a programmatic ``add`` and swallow any failure (the add tool's fire-and-forget mode)."""
        try:
            await self.add(content, MemoryAddOptions(stores=targets))
        except Exception:  # noqa: BLE001 - failures are logged in ``add``; swallow here.
            pass

    def _schedule_background(self, coroutine: Any) -> None:
        """Schedule a coroutine as a tracked background task."""
        task = asyncio.ensure_future(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with the agent.

        Wires up automatic extraction for any store configured with an
        ``ExtractionConfig``. A no-op when no store uses extraction.

        Extraction runs in the background. The synchronous ``Agent(...)`` entry
        point awaits :meth:`flush` after each invocation so writes persist;
        callers driving the agent through their own event loop should await
        :meth:`flush` at a shutdown boundary.
        """
        if len(self._extraction_stores) == 0:
            return

        coordinator = ExtractionCoordinator(self._extraction_stores, agent.model)
        self._coordinator = coordinator

        # Buffer every message so extraction has its own copy to save from.
        agent.add_hook(lambda event: coordinator.record(event.message), MessageAddedEvent)

        for store in self._extraction_stores:
            assert store.extraction is not None  # noqa: S101 - extraction stores always configure this.
            for trigger in _normalize_triggers(store.extraction.trigger):
                trigger.attach(ExtractionTriggerContext(agent=agent, fire=self._make_fire(coordinator, store)))

    @staticmethod
    def _make_fire(coordinator: ExtractionCoordinator, store: MemoryStore) -> Callable[[], None]:
        """Build a zero-arg ``fire`` callback bound to a specific store."""

        def fire() -> None:
            coordinator.schedule(store)

        return fire

    async def flush(self) -> None:
        """Save every store's remaining messages and wait for all saves to finish.

        A no-op when no store has extraction configured. Drains automatic
        extraction only; ``add_memory`` fire-and-forget writes are not awaited
        here.
        """
        if self._coordinator is not None:
            await self._coordinator.flush()
