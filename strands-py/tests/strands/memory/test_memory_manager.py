"""Tests for the MemoryManager primitive and the MemoryStore interface."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

import strands
from strands import Agent
from strands.hooks.events import AfterInvocationEvent
from strands.memory import (
    MemoryAddToolConfig,
    MemoryEntry,
    MemoryManager,
    MemoryStore,
    MemoryStoreError,
    MemoryToolConfig,
)
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.tools import AgentTool
from tests.fixtures.mocked_model_provider import MockedModelProvider


class FakeStore(MemoryStore):
    """In-memory store with spy-able search/add for tests."""

    def __init__(
        self,
        name: str,
        *,
        writable: bool = False,
        description: str | None = None,
        max_search_results: int | None = None,
        entries: list[MemoryEntry] | None = None,
        tools: list[AgentTool] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            writable=writable,
            description=description,
            max_search_results=max_search_results,
        )
        self._entries = entries if entries is not None else []
        self._tools = tools
        self.search = AsyncMock(return_value=self._entries)  # type: ignore[method-assign]
        if writable:
            self.add = AsyncMock(return_value=None)  # type: ignore[method-assign]

    async def search(self, query: str, *, max_search_results: int | None = None) -> list[MemoryEntry]:  # noqa: D102
        return self._entries

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:  # noqa: D102
        return None

    def get_tools(self) -> list[AgentTool]:  # noqa: D102
        return self._tools if self._tools is not None else []


class WritableNoAddStore(MemoryStore):
    """A store that claims writable=True but never overrides add (an error)."""

    async def search(self, query: str, *, max_search_results: int | None = None) -> list[MemoryEntry]:  # noqa: D102
        return []


def _named_tool(name: str) -> DecoratedFunctionTool:
    @strands.tool(name=name, description=f"test tool {name}")
    def _t() -> str:
        return "ok"

    return _t


def _search_tool(manager: MemoryManager) -> DecoratedFunctionTool:
    return next(t for t in manager.tools if t.tool_name == "search_memory")


def _add_tool(manager: MemoryManager) -> DecoratedFunctionTool:
    return next(t for t in manager.tools if t.tool_name == "add_memory")


async def _invoke(tool: DecoratedFunctionTool, alist, **tool_input: Any) -> dict[str, Any]:
    """Invoke a tool via its stream and return the final ToolResult."""
    events = await alist(tool.stream({"toolUseId": "t", "input": tool_input}, {}))
    return events[-1].tool_result


# --- Group A: constructor validation -------------------------------------------------


class TestConstructorValidation:
    def test_empty_stores_raises(self):
        with pytest.raises(ValueError, match="at least one store"):
            MemoryManager(stores=[])

    def test_creates_instance_with_valid_config(self):
        manager = MemoryManager(stores=[FakeStore("test")])
        assert manager.name == "strands.memory_manager"

    def test_duplicate_store_name_raises(self):
        with pytest.raises(ValueError, match="duplicate store name 'dup'"):
            MemoryManager(stores=[FakeStore("dup"), FakeStore("dup")])

    def test_writable_without_add_raises(self):
        with pytest.raises(ValueError, match="writable but does not implement add"):
            MemoryManager(stores=[WritableNoAddStore(name="broken", writable=True)])

    def test_add_tool_enabled_without_writable_store_raises(self):
        with pytest.raises(ValueError, match="no stores are writable"):
            MemoryManager(stores=[FakeStore("a")], add_tool_config=True)

    def test_add_tool_config_unknown_store_raises(self):
        with pytest.raises(ValueError, match="add_tool_config store 'nope' not found"):
            MemoryManager(
                stores=[FakeStore("a", writable=True)],
                add_tool_config=MemoryAddToolConfig(stores=["nope"]),
            )

    def test_add_tool_config_readonly_store_raises(self):
        with pytest.raises(ValueError, match="add_tool_config store 'readonly' is not writable"):
            MemoryManager(
                stores=[FakeStore("a", writable=True), FakeStore("readonly")],
                add_tool_config=MemoryAddToolConfig(stores=["readonly"]),
            )

    def test_add_tool_config_accepts_store_instances(self):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team], add_tool_config=MemoryAddToolConfig(stores=[personal]))
        assert [s.name for s in manager._add_tool_stores] == ["personal"]

    def test_add_tool_config_stray_instance_raises(self):
        configured = FakeStore("configured", writable=True)
        stray = FakeStore("stray", writable=True)
        with pytest.raises(ValueError, match="add_tool_config store 'stray' not found"):
            MemoryManager(stores=[configured], add_tool_config=MemoryAddToolConfig(stores=[stray]))


# --- Group B: tool registration & plugin wiring --------------------------------------


class TestTools:
    def test_registers_search_tool_by_default(self):
        manager = MemoryManager(stores=[FakeStore("test")])
        assert [t.tool_name for t in manager.tools] == ["search_memory"]

    def test_registers_add_tool_when_enabled(self):
        manager = MemoryManager(stores=[FakeStore("test", writable=True)], add_tool_config=True)
        assert [t.tool_name for t in manager.tools] == ["search_memory", "add_memory"]

    def test_no_add_tool_by_default(self):
        manager = MemoryManager(stores=[FakeStore("test", writable=True)])
        assert [t.tool_name for t in manager.tools] == ["search_memory"]

    def test_no_tools_when_both_disabled(self):
        manager = MemoryManager(
            stores=[FakeStore("test", writable=True)],
            search_tool_config=False,
            add_tool_config=False,
        )
        assert manager.tools == []

    def test_custom_tool_names(self):
        manager = MemoryManager(
            stores=[FakeStore("test", writable=True)],
            search_tool_config=MemoryToolConfig(name="recall"),
            add_tool_config=MemoryAddToolConfig(name="remember"),
        )
        assert [t.tool_name for t in manager.tools] == ["recall", "remember"]

    def test_search_tool_includes_store_descriptions(self):
        manager = MemoryManager(stores=[FakeStore("personal", description="User preferences")])
        desc = _search_tool(manager).tool_spec["description"]
        assert "personal: User preferences" in desc
        assert "target one or more memory stores by name" in desc

    def test_add_tool_includes_store_descriptions(self):
        manager = MemoryManager(
            stores=[FakeStore("notes", writable=True, description="Personal notes")],
            add_tool_config=True,
        )
        desc = _add_tool(manager).tool_spec["description"]
        assert "notes: Personal notes" in desc
        assert "target a specific store by name" in desc

    def test_drain_hook_discovered(self):
        manager = MemoryManager(stores=[FakeStore("test")])
        assert [getattr(h, "__name__", h) for h in manager.hooks] == ["_drain_pending_writes"]

    def test_search_tool_input_schema_props(self):
        manager = MemoryManager(stores=[FakeStore("test")])
        props = _search_tool(manager).tool_spec["inputSchema"]["json"]["properties"]
        assert set(props) == {"query", "max_search_results", "stores"}

    def test_add_tool_input_schema_props(self):
        manager = MemoryManager(stores=[FakeStore("test", writable=True)], add_tool_config=True)
        props = _add_tool(manager).tool_spec["inputSchema"]["json"]["properties"]
        assert set(props) == {"entries", "stores"}


class TestStoreProvidedTools:
    def test_store_tools_registered_via_init_agent(self):
        store = FakeStore("kb", tools=[_named_tool("kb_query")])
        agent = Agent(memory_manager=MemoryManager(stores=[store]))
        assert "kb_query" in agent.tool_names
        assert "search_memory" in agent.tool_names

    def test_store_tools_registered_even_with_no_manager_tools(self):
        store = FakeStore("kb", tools=[_named_tool("kb_query")])
        agent = Agent(memory_manager=MemoryManager(stores=[store], search_tool_config=False))
        assert "kb_query" in agent.tool_names
        assert "search_memory" not in agent.tool_names


# --- Group C: programmatic search() --------------------------------------------------


@pytest.mark.asyncio
class TestSearch:
    async def test_queries_all_stores_and_concatenates(self):
        a = FakeStore("a", entries=[MemoryEntry(content="fact one")])
        b = FakeStore("b", entries=[MemoryEntry(content="fact two")])
        manager = MemoryManager(stores=[a, b])

        results = await manager.search("query")
        assert results == [
            MemoryEntry(content="fact one", store_name="a"),
            MemoryEntry(content="fact two", store_name="b"),
        ]

    async def test_resolves_store_max_search_results_when_caller_omits(self):
        store = FakeStore("a", max_search_results=5)
        manager = MemoryManager(stores=[store])

        await manager.search("query")
        store.search.assert_called_once_with("query", max_search_results=5)

    async def test_forwards_explicit_max_search_results(self):
        store = FakeStore("a", max_search_results=5)
        manager = MemoryManager(stores=[store])

        await manager.search("query", max_search_results=2)
        store.search.assert_called_once_with("query", max_search_results=2)

    async def test_falls_back_to_sdk_default_limit(self):
        store = FakeStore("a")
        manager = MemoryManager(stores=[store])

        await manager.search("query")
        store.search.assert_called_once_with("query", max_search_results=3)

    async def test_filters_to_named_stores(self):
        personal = FakeStore("personal", entries=[MemoryEntry(content="personal fact")])
        team = FakeStore("team", entries=[MemoryEntry(content="team fact")])
        manager = MemoryManager(stores=[personal, team])

        results = await manager.search("query", stores=["personal"])
        assert results == [MemoryEntry(content="personal fact", store_name="personal")]
        team.search.assert_not_called()

    async def test_gracefully_handles_store_failures(self, caplog):
        failing = FakeStore("failing")
        failing.search = AsyncMock(side_effect=RuntimeError("network error"))
        ok = FakeStore("ok", entries=[MemoryEntry(content="fact")])
        manager = MemoryManager(stores=[failing, ok])

        with caplog.at_level(logging.WARNING):
            results = await manager.search("query")
        assert results == [MemoryEntry(content="fact", store_name="ok")]
        assert "failing" in caplog.text

    async def test_searches_all_when_stores_omitted(self):
        a = FakeStore("a", entries=[MemoryEntry(content="one")])
        b = FakeStore("b", entries=[MemoryEntry(content="two")])
        manager = MemoryManager(stores=[a, b])

        results = await manager.search("query")
        assert {e.store_name for e in results} == {"a", "b"}

    async def test_searches_none_when_stores_empty(self):
        a = FakeStore("a", entries=[MemoryEntry(content="one")])
        b = FakeStore("b", entries=[MemoryEntry(content="two")])
        manager = MemoryManager(stores=[a, b])

        results = await manager.search("query", stores=[])
        assert results == []
        a.search.assert_not_called()
        b.search.assert_not_called()

    async def test_unknown_named_store_raises(self):
        store = FakeStore("personal")
        manager = MemoryManager(stores=[store])

        with pytest.raises(ValueError, match="store 'nonexistent' not found"):
            await manager.search("query", stores=["nonexistent"])
        store.search.assert_not_called()


# --- Group D: programmatic add() -----------------------------------------------------


@pytest.mark.asyncio
class TestAdd:
    async def test_writes_to_all_writable_stores(self):
        a = FakeStore("a", writable=True)
        b = FakeStore("b", writable=True)
        manager = MemoryManager(stores=[a, b])

        await manager.add("user likes coffee")
        a.add.assert_called_once_with("user likes coffee", None)
        b.add.assert_called_once_with("user likes coffee", None)

    async def test_passes_metadata(self):
        store = FakeStore("a", writable=True)
        manager = MemoryManager(stores=[store])

        await manager.add("fact", metadata={"source": "user"})
        store.add.assert_called_once_with("fact", {"source": "user"})

    async def test_filters_to_named_stores(self):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team])

        await manager.add("pref", stores=["personal"])
        personal.add.assert_called_once_with("pref", None)
        team.add.assert_not_called()

    async def test_dedupes_duplicate_store_names(self):
        store = FakeStore("personal", writable=True)
        manager = MemoryManager(stores=[store])

        await manager.add("fact", stores=["personal", "personal"])
        store.add.assert_called_once()

    async def test_no_writable_store_matched_raises(self):
        manager = MemoryManager(stores=[FakeStore("a")])
        with pytest.raises(ValueError, match="no writable store matched"):
            await manager.add("fact")

    async def test_unknown_named_store_raises(self):
        manager = MemoryManager(stores=[FakeStore("a", writable=True)])
        with pytest.raises(ValueError, match="store 'nonexistent' not found"):
            await manager.add("fact", stores=["nonexistent"])

    async def test_readonly_named_store_raises(self):
        manager = MemoryManager(stores=[FakeStore("readonly")])
        with pytest.raises(ValueError, match="store 'readonly' is read-only"):
            await manager.add("fact", stores=["readonly"])

    async def test_partial_failure_raises_aggregate(self, caplog):
        failing = FakeStore("failing", writable=True)
        failing.add = AsyncMock(side_effect=RuntimeError("write error"))
        ok = FakeStore("ok", writable=True)
        manager = MemoryManager(stores=[failing, ok])

        with caplog.at_level(logging.WARNING), pytest.raises(MemoryStoreError, match="store writes failed: failing"):
            await manager.add("fact")
        ok.add.assert_called_once_with("fact", None)


# --- Group E: search_memory tool scoping --------------------------------------------


@pytest.mark.asyncio
class TestSearchToolScoping:
    async def test_searches_all_when_omitted(self, alist):
        personal = FakeStore("personal", entries=[MemoryEntry(content="p")])
        team = FakeStore("team", entries=[MemoryEntry(content="t")])
        manager = MemoryManager(stores=[personal, team])

        await _invoke(_search_tool(manager), alist, query="q")
        personal.search.assert_called()
        team.search.assert_called()

    async def test_empty_stores_treated_as_all(self, alist):
        personal = FakeStore("personal", entries=[MemoryEntry(content="p")])
        team = FakeStore("team", entries=[MemoryEntry(content="t")])
        manager = MemoryManager(stores=[personal, team])

        await _invoke(_search_tool(manager), alist, query="q", stores=[])
        personal.search.assert_called()
        team.search.assert_called()

    async def test_targets_only_requested_in_scope(self, alist):
        personal = FakeStore("personal", entries=[MemoryEntry(content="p")])
        team = FakeStore("team", entries=[MemoryEntry(content="t")])
        manager = MemoryManager(stores=[personal, team])

        await _invoke(_search_tool(manager), alist, query="q", stores=["personal"])
        personal.search.assert_called()
        team.search.assert_not_called()

    async def test_result_attributes_each_entry(self, alist):
        personal = FakeStore("personal", entries=[MemoryEntry(content="personal fact")])
        team = FakeStore("team", entries=[MemoryEntry(content="team fact")])
        manager = MemoryManager(stores=[personal, team])

        result = await _invoke(_search_tool(manager), alist, query="q")
        payload = json.loads(result["content"][0]["text"])
        assert payload == {
            "results": [
                {"content": "personal fact", "store_name": "personal"},
                {"content": "team fact", "store_name": "team"},
            ]
        }

    async def test_keeps_valid_warns_on_out_of_scope(self, alist, caplog):
        personal = FakeStore("personal", entries=[MemoryEntry(content="p")])
        team = FakeStore("team", entries=[MemoryEntry(content="t")])
        manager = MemoryManager(stores=[personal, team])

        with caplog.at_level(logging.WARNING):
            await _invoke(_search_tool(manager), alist, query="q", stores=["personal", "nonexistent"])
        personal.search.assert_called()
        team.search.assert_not_called()
        assert "nonexistent" in caplog.text

    async def test_all_out_of_scope_returns_error_result(self, alist):
        personal = FakeStore("personal", entries=[MemoryEntry(content="p")])
        manager = MemoryManager(stores=[personal])

        result = await _invoke(_search_tool(manager), alist, query="q", stores=["nonexistent"])
        assert result["status"] == "error"
        assert "none of the requested memory stores are available" in result["content"][0]["text"]
        personal.search.assert_not_called()

    async def test_underlying_failure_still_returns_partial_success(self, alist):
        failing = FakeStore("failing")
        failing.search = AsyncMock(side_effect=RuntimeError("boom"))
        ok = FakeStore("ok", entries=[MemoryEntry(content="fact")])
        manager = MemoryManager(stores=[failing, ok])

        result = await _invoke(_search_tool(manager), alist, query="q")
        assert result["status"] == "success"
        payload = json.loads(result["content"][0]["text"])
        assert payload == {"results": [{"content": "fact", "store_name": "ok"}]}


# --- Group F: add_memory tool scoping & write modes ---------------------------------


@pytest.mark.asyncio
class TestAddToolScoping:
    async def test_writes_all_writable_when_omitted(self, alist):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team], add_tool_config=True)

        await _invoke(_add_tool(manager), alist, entries=["fact"])
        personal.add.assert_called_once_with("fact", None)
        team.add.assert_called_once_with("fact", None)

    async def test_empty_stores_treated_as_all(self, alist):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team], add_tool_config=True)

        await _invoke(_add_tool(manager), alist, entries=["fact"], stores=[])
        personal.add.assert_called()
        team.add.assert_called()

    async def test_scoped_to_add_tool_config_stores(self, alist):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team], add_tool_config=MemoryAddToolConfig(stores=["personal"]))

        await _invoke(_add_tool(manager), alist, entries=["fact"])
        personal.add.assert_called_once_with("fact", None)
        team.add.assert_not_called()

    async def test_rejects_writable_store_excluded_from_allowlist(self, alist):
        personal = FakeStore("personal", writable=True)
        extraction_only = FakeStore("extraction-only", writable=True)
        manager = MemoryManager(
            stores=[personal, extraction_only],
            add_tool_config=MemoryAddToolConfig(stores=["personal"]),
        )

        result = await _invoke(_add_tool(manager), alist, entries=["fact"], stores=["extraction-only"])
        assert result["status"] == "error"
        assert "none of the requested memory stores are available" in result["content"][0]["text"]
        extraction_only.add.assert_not_called()

    async def test_excludes_readonly_stores_from_scope(self, alist):
        personal = FakeStore("personal", writable=True)
        readonly = FakeStore("readonly")
        manager = MemoryManager(stores=[personal, readonly], add_tool_config=True)

        result = await _invoke(_add_tool(manager), alist, entries=["fact"], stores=["readonly"])
        assert result["status"] == "error"
        assert "none of the requested memory stores are available" in result["content"][0]["text"]
        personal.add.assert_not_called()

    async def test_keeps_valid_warns_on_out_of_scope(self, alist, caplog):
        personal = FakeStore("personal", writable=True)
        team = FakeStore("team", writable=True)
        manager = MemoryManager(stores=[personal, team], add_tool_config=True)

        with caplog.at_level(logging.WARNING):
            await _invoke(_add_tool(manager), alist, entries=["fact"], stores=["personal", "nonexistent"])
        personal.add.assert_called_once_with("fact", None)
        team.add.assert_not_called()
        assert "nonexistent" in caplog.text

    async def test_all_out_of_scope_returns_error(self, alist):
        personal = FakeStore("personal", writable=True)
        manager = MemoryManager(stores=[personal], add_tool_config=True)

        result = await _invoke(_add_tool(manager), alist, entries=["fact"], stores=["nonexistent"])
        assert result["status"] == "error"
        assert "none of the requested memory stores are available" in result["content"][0]["text"]
        personal.add.assert_not_called()

    async def test_empty_entries_rejected(self, alist):
        personal = FakeStore("personal", writable=True)
        manager = MemoryManager(stores=[personal], add_tool_config=True)

        result = await _invoke(_add_tool(manager), alist, entries=[])
        assert result["status"] == "error"
        personal.add.assert_not_called()

    async def test_returns_stored_count_by_default(self, alist):
        store = FakeStore("notes", writable=True)
        manager = MemoryManager(stores=[store], add_tool_config=True)

        result = await _invoke(_add_tool(manager), alist, entries=["a", "b"])
        assert result["status"] == "success"
        assert json.loads(result["content"][0]["text"]) == {"stored": 2}

    async def test_failure_returns_error_with_concrete_reasons(self, alist):
        failing = FakeStore("failing", writable=True)
        failing.add = AsyncMock(side_effect=RuntimeError("write error"))
        manager = MemoryManager(stores=[failing], add_tool_config=True)

        result = await _invoke(_add_tool(manager), alist, entries=["a", "b"])
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "failed to add 2 of 2 entries" in text
        assert "write error" in text

    async def test_wait_for_writes_false_returns_accepted(self, alist):
        store = FakeStore("notes", writable=True)
        manager = MemoryManager(stores=[store], add_tool_config=MemoryAddToolConfig(wait_for_writes=False))

        result = await _invoke(_add_tool(manager), alist, entries=["a", "b"])
        assert result["status"] == "success"
        assert json.loads(result["content"][0]["text"]) == {"accepted": 2}

        # Drain dispatched writes and confirm they landed.
        await manager._drain_pending_writes(AfterInvocationEvent(agent=None))
        assert store.add.call_count == 2

    async def test_wait_for_writes_false_swallows_failures(self, alist, caplog):
        failing = FakeStore("failing", writable=True)
        failing.add = AsyncMock(side_effect=RuntimeError("write error"))
        manager = MemoryManager(stores=[failing], add_tool_config=MemoryAddToolConfig(wait_for_writes=False))

        with caplog.at_level(logging.WARNING):
            result = await _invoke(_add_tool(manager), alist, entries=["a", "b"])
            assert json.loads(result["content"][0]["text"]) == {"accepted": 2}
            await manager._drain_pending_writes(AfterInvocationEvent(agent=None))
        assert "fire-and-forget memory write failed" in caplog.text


# --- Group G: Agent integration ------------------------------------------------------


class TestAgentIntegration:
    def test_registers_memory_tools_on_agent(self):
        agent = Agent(memory_manager=MemoryManager(stores=[FakeStore("test", writable=True)], add_tool_config=True))
        assert "search_memory" in agent.tool_names
        assert "add_memory" in agent.tool_names

    def test_passes_through_instance_unchanged(self):
        manager = MemoryManager(stores=[FakeStore("test")])
        agent = Agent(memory_manager=manager)
        assert agent.memory_manager is manager

    def test_memory_manager_none_when_not_configured(self):
        agent = Agent()
        assert agent.memory_manager is None

    def test_name_collision_raises(self):
        # Two managers register under the same plugin name on one agent.
        manager = MemoryManager(stores=[FakeStore("a")])
        with pytest.raises(ValueError, match="already registered"):
            Agent(memory_manager=manager, plugins=[MemoryManager(stores=[FakeStore("b")])])


@pytest.mark.asyncio
class TestDrainOnInvocation:
    async def test_pending_writes_drained_after_invocation(self, alist):
        store = FakeStore("notes", writable=True)
        manager = MemoryManager(stores=[store], add_tool_config=MemoryAddToolConfig(wait_for_writes=False))
        tool_use = {"name": "add_memory", "toolUseId": "1", "input": {"entries": ["remember this"]}}
        model = MockedModelProvider(
            [
                {"role": "assistant", "content": [{"toolUse": tool_use}]},
                {"role": "assistant", "content": [{"text": "done"}]},
            ]
        )
        agent = Agent(model=model, memory_manager=manager)

        await agent.invoke_async("save a memory")

        # The drain hook on AfterInvocationEvent guarantees the write completed.
        assert manager._pending_writes == set()
        store.add.assert_called_once_with("remember this", None)


# --- Group H: types & edge cases -----------------------------------------------------


class TestTypes:
    def test_memory_entry_defaults(self):
        entry = MemoryEntry(content="x")
        assert entry.store_name is None
        assert entry.metadata is None

    @pytest.mark.asyncio
    async def test_memory_store_add_default_raises(self):
        store = FakeStore("ro")

        class BareStore(MemoryStore):
            async def search(self, query, *, max_search_results=None):
                return []

        bare = BareStore(name="bare")
        with pytest.raises(NotImplementedError, match="store 'bare' does not implement add"):
            await bare.add("x")
        # Sanity: Fake.search still works.
        assert await store.search("q") == []

    def test_memory_store_error_carries_errors(self):
        err = MemoryStoreError("boom", errors=[ValueError("a"), RuntimeError("b")])
        assert str(err) == "boom"
        assert len(err.errors) == 2
