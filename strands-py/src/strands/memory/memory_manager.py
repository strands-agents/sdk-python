"""Cross-session memory retrieval and storage for agents.

This module is a Python port of the TypeScript ``memory/memory-manager.ts``. The
:class:`MemoryManager` manages one or more :class:`~strands.memory.types.MemoryStore`
backends and exposes ``search_memory`` / ``add_memory`` tools for agent-driven
recall and persistence, alongside any tools the stores themselves provide via
:meth:`~strands.memory.types.MemoryStore.get_tools`.

As a :class:`~strands.plugins.plugin.Plugin`, the manager builds its tools at
construction (exposed via the ``tools`` property) and wires automatic extraction
in :meth:`MemoryManager.init_agent` for any store configured with an
``ExtractionConfig``.

All public field and method names use ``snake_case`` per the requirements naming
convention mapping (e.g. ``maxSearchResults`` -> ``max_search_results``,
``addMessages`` -> ``add_messages``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..hooks.events import AfterInvocationEvent, MessageAddedEvent
from ..hooks.registry import HookOrder
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

# Default maximum results per store when neither the caller nor the store
# specifies one. Resolved by the ``MemoryManager``.
DEFAULT_MAX_SEARCH_RESULTS = 3


def _normalize_triggers(trigger: ExtractionTrigger | list[ExtractionTrigger]) -> list[ExtractionTrigger]:
    """Normalize a store's ``trigger`` field (a single trigger or a list) to a list.

    Args:
        trigger: A single trigger or a list of triggers.

    Returns:
        The triggers as a list.
    """
    return list(trigger) if isinstance(trigger, list) else [trigger]


def _flatten_reasons(reasons: list[BaseException]) -> list[BaseException]:
    """Flatten nested aggregate errors so the leaves are concrete reasons.

    Any ``AggregateMemoryError`` in ``reasons`` is replaced by its own
    (recursively flattened) ``errors``, so the result holds concrete underlying
    errors rather than aggregates-of-aggregates.

    Args:
        reasons: The exceptions to flatten.

    Returns:
        A flat list of concrete leaf exceptions.
    """
    flattened: list[BaseException] = []
    for reason in reasons:
        if isinstance(reason, AggregateMemoryError):
            flattened.extend(_flatten_reasons(reason.errors))
        else:
            flattened.append(reason)
    return flattened


class MemoryManager(Plugin):
    """Provides cross-session memory retrieval and storage for agents.

    Manages one or more :class:`~strands.memory.types.MemoryStore` backends,
    exposing ``search_memory`` and ``add_memory`` tools for agent-driven recall
    and persistence. Any tools the stores themselves provide (via
    :meth:`~strands.memory.types.MemoryStore.get_tools`) are registered alongside
    these.

    When driving the agent through the synchronous ``Agent(...)`` entry point, set
    ``flush_on_invocation_end=True`` so extraction writes persist across its
    per-invocation event loop.

    Example:
        ```python
        from strands import Agent
        from strands.memory import MemoryManager

        # Synchronous Agent(...) entry point: enable flush_on_invocation_end so
        # extraction writes persist across each invocation's event loop.
        memory_manager = MemoryManager(stores=[my_store], flush_on_invocation_end=True)
        agent = Agent(model=model, plugins=[memory_manager])
        agent("Remember I prefer dark mode")

        # search/add/flush are coroutines; await them from async code.
        results = await memory_manager.search("user preferences")
        ```
    """

    name = "strands:memory-manager"

    def __init__(
        self,
        stores: list[MemoryStore],
        search_tool_config: MemoryToolConfig | bool = True,
        add_tool_config: MemoryAddToolConfig | bool = False,
        flush_on_invocation_end: bool = False,
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
            flush_on_invocation_end: When True, await pending extraction writes at
                the end of each agent invocation. Enable this when driving the
                agent through the synchronous ``Agent(...)`` entry point, whose
                per-invocation event loop would otherwise cancel in-flight
                background saves. Defaults to False (fire-and-forget).

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
                # Each extraction shape needs its matching write sink. An extractor
                # produces discrete entries written via `add`; without an extractor
                # the raw message batch goes to `add_messages`.
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
            # The `add_memory` tool writes via `add` (not `add_messages`), so it
            # needs an `add`-capable store.
            if len(self._add_stores) == 0:
                raise ValueError("MemoryManager: add_tool_config is enabled but no writable stores implement add")
            resolved_config = (
                add_tool_config if isinstance(add_tool_config, MemoryAddToolConfig) else MemoryAddToolConfig()
            )
            self._add_tool_config = resolved_config
            self._add_tool_stores = self._resolve_add_tool_stores(resolved_config)

        # Background fire-and-forget tasks (e.g. the add tool's non-blocking
        # writes), retained so they are not garbage collected mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

        # Background extraction coordinator, created in ``init_agent`` when
        # extraction is configured.
        self._coordinator: ExtractionCoordinator | None = None

        self._flush_on_invocation_end = flush_on_invocation_end

        # Build the manager's tools now and surface them via the ``tools``
        # property (which the plugin registry reads to register them).
        self._memory_tools: list[AgentTool] = self._build_tools()

    def _resolve_add_tool_stores(self, tool_config: MemoryAddToolConfig) -> list[MemoryStore]:
        """Resolve the writable stores the ``add_memory`` tool may write to.

        When ``stores`` is given, each entry (a store name or a
        :class:`~strands.memory.types.MemoryStore` instance) must resolve by name
        to a configured, ``add``-capable writable store. Omitted means all such
        stores.

        Args:
            tool_config: The add tool configuration.

        Returns:
            The allowlist of stores the add tool may write to.

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

        Includes the manager's own ``search_memory`` / ``add_memory`` tools (per
        their config) plus any tools the configured stores expose via
        :meth:`~strands.memory.types.MemoryStore.get_tools`, in store order.

        Returns:
            The tools to register with the agent.
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

        Widens the base :class:`~strands.plugins.plugin.Plugin` annotation
        (``list[DecoratedFunctionTool]``) because a store's ``get_tools`` may
        contribute any :class:`~strands.types.tools.AgentTool`.
        """
        return list(self._memory_tools)

    async def search(self, query: str, options: MemorySearchOptions | None = None) -> list[MemoryEntry]:
        """Search stores for entries matching the query.

        Unscoped: has full access to all configured stores. When
        ``options.stores`` is omitted, all stores are searched; tool-level store
        scoping is applied by the search tool callback.

        Args:
            query: The search query string.
            options: Optional max results (forwarded to all stores) and store
                name filter.

        Returns:
            Memory entries from matching stores, each attributed to its store via
            ``store_name``, concatenated in target order.

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

        Unscoped: has full access to all configured writable stores; tool-level
        scoping is applied by the add tool callback. Target stores are validated
        first (an unknown or read-only named store raises), then writes are
        awaited concurrently: per-store failures are logged and surfaced as an
        :class:`~strands.types.exceptions.AggregateMemoryError`.

        Args:
            content: The text content to add.
            options: Optional metadata and store name filter.

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

        - Omitting ``requested`` (``None`` or empty) targets all scoped stores.
        - In-scope names are kept; out-of-scope names are dropped with a warning.
        - When every requested name is out of scope, raises so the model receives
          an actionable error.

        Args:
            scoped_names: Store names available to this tool.
            requested: Store names the model asked for, if any.

        Returns:
            A non-empty list of in-scope store names to target.

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
        """Build the ``search_memory`` tool.

        Args:
            config: The search tool configuration.

        Returns:
            The search tool.
        """
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
        """Build the ``add_memory`` tool.

        Args:
            config: The add tool configuration.
            stores: The writable stores this tool may write to.

        Returns:
            The add tool.
        """
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
            # The Python @tool validation model does not enforce the advertised
            # JSON-schema ``minItems``, so guard against an empty batch here.
            if not entries:
                raise ValueError("MemoryManager: add_memory requires at least one entry")

            targets = self._resolve_tool_targets(scoped_names, stores)

            if not wait_for_writes:
                # Fire-and-forget: dispatch the writes without awaiting so the
                # agent loop isn't blocked. ``add`` logs per-store failures;
                # swallow the rejection so it isn't an unhandled exception.
                for content in entries:
                    self._schedule_background(self._add_swallow(content, targets))
                return {"accepted": len(entries)}

            # Await mode: surface failures to the model with concrete reasons
            # (not nested aggregate errors).
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
        """Run a programmatic ``add`` and swallow any failure.

        Used by the add tool's fire-and-forget mode: ``add`` already logs
        per-store failures, so the raised aggregate is intentionally ignored here
        to avoid an unhandled background exception.

        Args:
            content: The content to add.
            targets: The resolved target store names.
        """
        try:
            await self.add(content, MemoryAddOptions(stores=targets))
        except Exception:  # noqa: BLE001 - failures are logged in ``add``; swallow here.
            pass

    def _schedule_background(self, coroutine: Any) -> None:
        """Schedule a coroutine as a tracked background task.

        Args:
            coroutine: The coroutine to run in the background.
        """
        task = asyncio.ensure_future(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with the agent.

        Wires up automatic extraction for any store configured with an
        ``ExtractionConfig``: buffers conversation messages and attaches each
        store's triggers. A no-op when no store uses extraction.

        Args:
            agent: The agent this plugin is being attached to.
        """
        if len(self._extraction_stores) == 0:
            return

        coordinator = ExtractionCoordinator(self._extraction_stores, agent.model)
        self._coordinator = coordinator

        # Buffer every message the agent adds, so extraction has its own copy to
        # save from.
        agent.add_hook(lambda event: coordinator.record(event.message), MessageAddedEvent)

        for store in self._extraction_stores:
            assert store.extraction is not None  # noqa: S101 - extraction stores always configure this.
            for trigger in _normalize_triggers(store.extraction.trigger):
                trigger.attach(ExtractionTriggerContext(agent=agent, fire=self._make_fire(coordinator, store)))

        if self._flush_on_invocation_end:
            agent.add_hook(self._flush_after_invocation, AfterInvocationEvent, order=HookOrder.SDK_LAST)
        else:
            logger.warning(
                "flush_on_invocation_end=<False> | background extraction is lost if the event loop closes "
                "before it finishes (e.g. the synchronous Agent(...) entry point); safe to ignore if you "
                "await MemoryManager.flush() at a shutdown boundary or enable flush_on_invocation_end."
            )

    async def _flush_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Await pending extraction writes at the end of an agent invocation."""
        await self.flush()

    @staticmethod
    def _make_fire(coordinator: ExtractionCoordinator, store: MemoryStore) -> Callable[[], None]:
        """Build a zero-arg ``fire`` callback bound to a specific store.

        Binds the store here (rather than in a loop-body lambda) to avoid the
        late-binding closure pitfall and to keep a clean ``Callable[[], None]``.
        """

        def fire() -> None:
            coordinator.schedule(store)

        return fire

    async def flush(self) -> None:
        """Save every store's remaining messages and wait for all saves to finish.

        A no-op when no store has extraction configured. Call this at a boundary
        you control (typically app shutdown) so the most recent turn is not lost.

        Drains automatic extraction only; ``add_memory`` fire-and-forget writes
        (``wait_for_writes=False``) are dispatched but not awaited here.
        """
        if self._coordinator is not None:
            await self._coordinator.flush()
