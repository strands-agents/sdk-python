"""Tests for :class:`~strands.memory.stores.graphiti.GraphitiMemoryStore`.

Graphiti is an optional dependency that is not installed in the unit-test
environment, so these tests never import ``graphiti-core``. Instead they:

- mock the ``Graphiti`` client (an :class:`~unittest.mock.AsyncMock` whose
  ``search`` / ``add_episode`` are programmed and inspected), so no graph
  database is required; and
- install a fake ``graphiti_core.nodes`` module (carrying an ``EpisodeType``
  enum) for the writes that resolve the episode type lazily, and assert the bare
  :class:`ImportError` path when that module is absent.

Search results are faked as ``EntityEdge``-shaped objects
(:func:`_edge`) carrying the ``fact`` and the bi-temporal timestamps the store
maps into entry metadata.
"""

from __future__ import annotations

import builtins
import sys
from datetime import datetime, timezone
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from strands.memory import GraphitiMemoryStore as ExportedGraphitiMemoryStore
from strands.memory.memory_manager import MemoryManager
from strands.memory.stores.graphiti import (
    DEFAULT_MAX_SEARCH_RESULTS,
    GraphitiMemoryStore,
)
from strands.memory.types import (
    MemoryEntry,
    MemorySearchOptions,
    SearchOptions,
    _has_method,
    _has_write_sink,
)

# --------------------------------------------------------------------------- #
# Test fakes / helpers
# --------------------------------------------------------------------------- #


class _FakeEpisodeType(Enum):
    """Stand-in for ``graphiti_core.nodes.EpisodeType`` keyed by member name."""

    message = "message"
    json = "json"
    text = "text"


@pytest.fixture
def fake_graphiti_nodes() -> Any:
    """Install a fake ``graphiti_core.nodes`` module exposing ``EpisodeType``.

    The store imports ``graphiti_core.nodes`` lazily on a write; without the real package installed
    the import would fail, so we inject a minimal stand-in for the duration of a test.
    """
    package = ModuleType("graphiti_core")
    nodes = ModuleType("graphiti_core.nodes")
    nodes.EpisodeType = _FakeEpisodeType  # type: ignore[attr-defined]
    package.nodes = nodes  # type: ignore[attr-defined]

    sys.modules["graphiti_core"] = package
    sys.modules["graphiti_core.nodes"] = nodes
    try:
        yield nodes
    finally:
        sys.modules.pop("graphiti_core.nodes", None)
        sys.modules.pop("graphiti_core", None)


def _edge(
    *,
    fact: str,
    uuid: str = "edge-uuid",
    name: str = "RELATES_TO",
    group_id: str = "g1",
    source_node_uuid: str = "src",
    target_node_uuid: str = "tgt",
    episodes: list[str] | None = None,
    created_at: datetime | None = None,
    expired_at: datetime | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
) -> SimpleNamespace:
    """Build a fake Graphiti ``EntityEdge`` carrying ``fact`` and bi-temporal timestamps."""
    return SimpleNamespace(
        fact=fact,
        uuid=uuid,
        name=name,
        group_id=group_id,
        source_node_uuid=source_node_uuid,
        target_node_uuid=target_node_uuid,
        episodes=episodes if episodes is not None else ["ep-1"],
        created_at=created_at,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def _client(*, search_result: list[Any] | None = None, add_result: Any = None) -> AsyncMock:
    """Build a mock ``Graphiti`` client with programmed ``search`` / ``add_episode``."""
    client = AsyncMock()
    client.search = AsyncMock(return_value=search_result if search_result is not None else [])
    client.add_episode = AsyncMock(return_value=add_result if add_result is not None else SimpleNamespace())
    return client


def _store(client: AsyncMock | None = None, **kwargs: Any) -> GraphitiMemoryStore:
    """Build a store with sensible defaults over a mock client."""
    params: dict[str, Any] = {"name": "graph", "writable": True}
    params.update(kwargs)
    return GraphitiMemoryStore(client=client if client is not None else _client(), **params)


# --------------------------------------------------------------------------- #
# Construction / identity
# --------------------------------------------------------------------------- #


def test_exposes_config_fields_as_attributes() -> None:
    store = GraphitiMemoryStore(
        client=_client(),
        name="graph",
        description="a graph",
        max_search_results=7,
        writable=True,
        extraction=True,
    )

    assert store.name == "graph"
    assert store.description == "a graph"
    assert store.max_search_results == 7
    assert store.writable is True
    assert store.extraction is True


def test_defaults_are_read_only_and_unconfigured() -> None:
    store = GraphitiMemoryStore(client=_client(), name="graph")

    assert store.writable is False
    assert store.description is None
    assert store.max_search_results is None
    assert store.extraction is None


def test_rejects_non_positive_max_search_results() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        GraphitiMemoryStore(client=_client(), name="graph", max_search_results=0)


def test_exported_from_memory_package() -> None:
    assert ExportedGraphitiMemoryStore is GraphitiMemoryStore


# --------------------------------------------------------------------------- #
# Optional-method detection (Protocol conformance)
# --------------------------------------------------------------------------- #


def test_detects_write_sinks_via_has_method() -> None:
    store = _store()

    # ``_has_method`` inspects ``type(store)``; concrete methods (not the Protocol stubs) count.
    assert _has_method(store, "add") is True
    assert _has_method(store, "add_messages") is True
    assert _has_write_sink(store) is True
    # ``get_tools`` is intentionally not implemented, so the manager won't call it.
    assert _has_method(store, "get_tools") is False


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_search_maps_edges_to_entries() -> None:
    client = _client(search_result=[_edge(fact="user prefers dark mode"), _edge(fact="user is in Berlin")])
    store = _store(client)

    entries = await store.search("preferences")

    assert [entry.content for entry in entries] == ["user prefers dark mode", "user is in Berlin"]
    assert all(isinstance(entry, MemoryEntry) for entry in entries)


@pytest.mark.asyncio
async def test_search_surfaces_identifiers_and_bitemporal_metadata() -> None:
    created = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    valid = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    client = _client(
        search_result=[
            _edge(
                fact="user moved to Berlin",
                uuid="e1",
                name="MOVED_TO",
                group_id="tenant-a",
                source_node_uuid="user",
                target_node_uuid="berlin",
                episodes=["ep-9"],
                created_at=created,
                valid_at=valid,
            )
        ]
    )
    store = _store(client)

    [entry] = await store.search("where does the user live")

    assert entry.metadata == {
        "uuid": "e1",
        "name": "MOVED_TO",
        "group_id": "tenant-a",
        "source_node_uuid": "user",
        "target_node_uuid": "berlin",
        "episodes": ["ep-9"],
        "created_at": created.isoformat(),
        "valid_at": valid.isoformat(),
    }
    # ``None`` timestamps (expired_at / invalid_at here) are dropped, not rendered.
    assert "expired_at" not in entry.metadata
    assert "invalid_at" not in entry.metadata


@pytest.mark.asyncio
async def test_search_does_not_set_store_name() -> None:
    # Attribution is the manager's job; the store leaves ``store_name`` unset.
    client = _client(search_result=[_edge(fact="x")])
    store = _store(client)

    [entry] = await store.search("q")

    assert entry.store_name is None


@pytest.mark.asyncio
async def test_search_uses_default_limit_when_unspecified() -> None:
    client = _client()
    store = _store(client, max_search_results=None)

    await store.search("q")

    _, kwargs = client.search.call_args
    assert kwargs["num_results"] == DEFAULT_MAX_SEARCH_RESULTS


@pytest.mark.asyncio
async def test_search_uses_store_default_limit() -> None:
    client = _client()
    store = _store(client, max_search_results=5)

    await store.search("q")

    _, kwargs = client.search.call_args
    assert kwargs["num_results"] == 5


@pytest.mark.asyncio
async def test_search_caller_limit_overrides_store_default() -> None:
    client = _client()
    store = _store(client, max_search_results=5)

    await store.search("q", SearchOptions(max_search_results=2))

    _, kwargs = client.search.call_args
    assert kwargs["num_results"] == 2


@pytest.mark.asyncio
async def test_search_rejects_non_positive_caller_limit() -> None:
    store = _store()

    with pytest.raises(ValueError, match="at least 1"):
        await store.search("q", SearchOptions(max_search_results=0))


@pytest.mark.asyncio
async def test_search_scopes_by_group_id() -> None:
    client = _client()
    store = _store(client, group_id="tenant-a")

    await store.search("q")

    _, kwargs = client.search.call_args
    assert kwargs["group_ids"] == ["tenant-a"]


@pytest.mark.asyncio
async def test_search_omits_group_ids_when_unscoped() -> None:
    client = _client()
    store = _store(client, group_id=None)

    await store.search("q")

    _, kwargs = client.search.call_args
    assert kwargs["group_ids"] is None


@pytest.mark.asyncio
async def test_search_treats_empty_group_id_as_unscoped() -> None:
    # An empty group id would scope search to a nonexistent group; normalize it to unscoped.
    client = _client()
    store = _store(client, group_id="")

    await store.search("q")

    _, kwargs = client.search.call_args
    assert kwargs["group_ids"] is None


@pytest.mark.asyncio
async def test_search_passes_query_positionally() -> None:
    client = _client()
    store = _store(client)

    await store.search("find this")

    args, _ = client.search.call_args
    assert args[0] == "find this"


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_writes_text_episode(fake_graphiti_nodes: Any) -> None:
    result = SimpleNamespace(episode="ok")
    client = _client(add_result=result)
    store = _store(client, group_id="tenant-a")

    returned = await store.add("user prefers dark mode")

    assert returned is result
    _, kwargs = client.add_episode.call_args
    assert kwargs["episode_body"] == "user prefers dark mode"
    assert kwargs["source"] is _FakeEpisodeType.text
    assert kwargs["group_id"] == "tenant-a"
    assert isinstance(kwargs["reference_time"], datetime)
    assert kwargs["source_description"] == "strands-agent-memory"


@pytest.mark.asyncio
async def test_add_honours_metadata_episode_parameters(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client, group_id="tenant-a")
    reference = datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)

    await store.add(
        "fact",
        metadata={
            "name": "custom-name",
            "source_description": "import job",
            "reference_time": reference,
            "group_id": "tenant-b",
            "uuid": "stable-id",
            "ignored": "dropped",
        },
    )

    _, kwargs = client.add_episode.call_args
    assert kwargs["name"] == "custom-name"
    assert kwargs["source_description"] == "import job"
    assert kwargs["reference_time"] == reference
    assert kwargs["group_id"] == "tenant-b"  # per-write override beats the store namespace
    assert kwargs["uuid"] == "stable-id"
    assert "ignored" not in kwargs


@pytest.mark.asyncio
async def test_add_accepts_iso_reference_time(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)

    await store.add("fact", metadata={"reference_time": "2026-05-01T09:30:00+00:00"})

    _, kwargs = client.add_episode.call_args
    assert kwargs["reference_time"] == datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_add_makes_naive_reference_time_utc_aware(fake_graphiti_nodes: Any) -> None:
    # Graphiti's bi-temporal layer compares against tz-aware UTC, so a naive input must be made
    # aware (interpreted as UTC) rather than passed through naive.
    client = _client()
    store = _store(client)

    await store.add("fact", metadata={"reference_time": datetime(2026, 5, 1, 9, 30)})

    _, kwargs = client.add_episode.call_args
    assert kwargs["reference_time"] == datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)
    assert kwargs["reference_time"].tzinfo is not None


@pytest.mark.asyncio
async def test_add_converts_aware_reference_time_to_utc(fake_graphiti_nodes: Any) -> None:
    from datetime import timedelta

    client = _client()
    store = _store(client)
    plus_two = timezone(timedelta(hours=2))

    await store.add("fact", metadata={"reference_time": datetime(2026, 5, 1, 11, 30, tzinfo=plus_two)})

    _, kwargs = client.add_episode.call_args
    assert kwargs["reference_time"] == datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_add_defaults_reference_time_to_utc_now(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)

    await store.add("fact")

    _, kwargs = client.add_episode.call_args
    assert kwargs["reference_time"].tzinfo is not None


@pytest.mark.asyncio
async def test_add_empty_metadata_group_id_falls_back_to_store(fake_graphiti_nodes: Any) -> None:
    # An empty per-write group id is meaningless; fall back to the store's namespace.
    client = _client()
    store = _store(client, group_id="tenant-a")

    await store.add("fact", metadata={"group_id": ""})

    _, kwargs = client.add_episode.call_args
    assert kwargs["group_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_add_omits_uuid_when_not_supplied(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)

    await store.add("fact")

    _, kwargs = client.add_episode.call_args
    assert "uuid" not in kwargs


@pytest.mark.asyncio
async def test_add_rejects_when_not_writable(fake_graphiti_nodes: Any) -> None:
    store = _store(writable=False)

    with pytest.raises(ValueError, match="not writable"):
        await store.add("fact")


@pytest.mark.asyncio
async def test_add_rejects_empty_content(fake_graphiti_nodes: Any) -> None:
    store = _store()

    with pytest.raises(ValueError, match="must not be empty"):
        await store.add("   ")


@pytest.mark.asyncio
async def test_add_raises_helpful_error_when_graphiti_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure ``graphiti_core`` is absent and its import fails, simulating a missing optional dep.
    monkeypatch.delitem(sys.modules, "graphiti_core", raising=False)
    monkeypatch.delitem(sys.modules, "graphiti_core.nodes", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "graphiti_core.nodes" or name == "graphiti_core":
            raise ImportError("No module named 'graphiti_core'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    store = _store()

    with pytest.raises(ImportError, match=r"strands-agents\[graphiti\]"):
        await store.add("fact")


# --------------------------------------------------------------------------- #
# add_messages
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_add_messages_writes_message_episode_transcript(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)
    messages = [
        {"role": "user", "content": [{"text": "I love hiking"}]},
        {"role": "assistant", "content": [{"text": "Noted!"}]},
    ]

    await store.add_messages(messages)  # type: ignore[arg-type]

    _, kwargs = client.add_episode.call_args
    assert kwargs["source"] is _FakeEpisodeType.message
    assert kwargs["episode_body"] == "user: I love hiking\nassistant: Noted!"


@pytest.mark.asyncio
async def test_add_messages_skips_non_text_blocks(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)
    messages = [
        {"role": "user", "content": [{"text": "keep this"}, {"toolUse": {"name": "x"}}]},
    ]

    await store.add_messages(messages)  # type: ignore[arg-type]

    _, kwargs = client.add_episode.call_args
    assert kwargs["episode_body"] == "user: keep this"


@pytest.mark.asyncio
async def test_add_messages_returns_none_for_empty_transcript(fake_graphiti_nodes: Any) -> None:
    client = _client()
    store = _store(client)
    messages = [{"role": "user", "content": [{"toolResult": {"content": []}}]}]

    result = await store.add_messages(messages)  # type: ignore[arg-type]

    assert result is None
    client.add_episode.assert_not_called()


@pytest.mark.asyncio
async def test_add_messages_rejects_when_not_writable(fake_graphiti_nodes: Any) -> None:
    store = _store(writable=False)
    messages = [{"role": "user", "content": [{"text": "hi"}]}]

    with pytest.raises(ValueError, match="not writable"):
        await store.add_messages(messages)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Integration with MemoryManager (Protocol conformance end to end)
# --------------------------------------------------------------------------- #


def test_manager_accepts_writable_store_without_error() -> None:
    # A writable store must expose a write sink; the manager validates this at construction.
    store = _store(name="graph", writable=True)
    manager = MemoryManager(stores=[store], add_tool_config=True)

    tool_names = {agent_tool.tool_name for agent_tool in manager.tools}
    assert "search_memory" in tool_names
    assert "add_memory" in tool_names


@pytest.mark.asyncio
async def test_manager_search_attributes_entries_to_store() -> None:
    client = _client(search_result=[_edge(fact="user prefers dark mode")])
    store = _store(client, name="graph")
    manager = MemoryManager(stores=[store], injection=False)

    results = await manager.search("preferences", MemorySearchOptions(max_search_results=4))

    assert [entry.content for entry in results] == ["user prefers dark mode"]
    assert results[0].store_name == "graph"
    # The manager forwards its per-store limit down to the store, which forwards it to Graphiti.
    _, kwargs = client.search.call_args
    assert kwargs["num_results"] == 4
