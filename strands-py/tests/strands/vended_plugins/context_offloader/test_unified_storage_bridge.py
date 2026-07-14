"""Tests for the ContextOffloader bridge onto the unified ``strands.storage`` interface.

Covers the pieces added to wire the offloader onto a unified ``Storage`` backend
(write/read/delete/list), matching the strands-ts integration:

- content-type framing round-trip,
- legacy-vs-unified backend detection,
- store/retrieve adapters,
- end-to-end offload + retrieve through a unified backend,
- plugin-driven eviction (unified backends have no built-in eviction),
- ``evict_after_cycles`` validation, and
- forwarding the eviction window to a legacy ``InMemoryStorage``.
"""

import json
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from strands.hooks.events import AfterToolCallEvent, BeforeModelCallEvent
from strands.storage import InMemoryStorage as UnifiedInMemoryStorage
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage
from strands.vended_plugins.context_offloader.plugin import (
    _frame_content,
    _is_offloader_storage,
    _retrieve_content,
    _store_content,
    _unframe_content,
)


async def _count_tokens(messages, **kwargs):
    """Heuristic token counter for tests: chars / 4 over tool-result text/json."""
    total = 0
    for msg in messages:
        for block in msg.get("content", []):
            if "toolResult" in block:
                for content in block["toolResult"].get("content", []):
                    if "text" in content:
                        total += math.ceil(len(content["text"]) / 4)
                    elif "json" in content:
                        total += math.ceil(len(json.dumps(content["json"])) / 4)
    return total


def _agent(cycle_count=0):
    agent = MagicMock()
    agent.model.count_tokens = AsyncMock(side_effect=_count_tokens)
    agent.event_loop_metrics.cycle_count = cycle_count
    return agent


def _after_tool_event(agent, content, tool_use_id="tool_123"):
    result = {"toolUseId": tool_use_id, "status": "success", "content": content}
    tool_use = {"toolUseId": tool_use_id, "name": "test_tool", "input": {}}
    return AfterToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use=tool_use,
        invocation_state={},
        result=result,
        cancel_message=None,
    )


def _before_model_event(agent, cycle_count):
    agent.event_loop_metrics.cycle_count = cycle_count
    return BeforeModelCallEvent(agent=agent, invocation_state={})


class TestFraming:
    @pytest.mark.parametrize(
        "content, content_type",
        [
            (b"hello world", "text/plain"),
            (b'{"a": 1}', "application/json"),
            (b"\x89PNG\r\n\x1a\n\x00\xff", "image/png"),
            (b"", "application/octet-stream"),
            (bytes(range(256)), "application/pdf"),
        ],
    )
    def test_round_trips_content_and_type(self, content, content_type):
        content_out, type_out = _unframe_content(_frame_content(content, content_type))
        assert content_out == content
        assert type_out == content_type

    def test_layout_is_big_endian_length_prefixed(self):
        frame = _frame_content(b"BODY", "text/plain")
        assert frame[:2] == len("text/plain").to_bytes(2, "big")
        assert frame[2:].startswith(b"text/plain")
        assert frame.endswith(b"BODY")


class TestStorageDetection:
    def test_legacy_offloader_storage_detected(self):
        assert _is_offloader_storage(InMemoryStorage()) is True

    def test_unified_storage_not_detected(self):
        assert _is_offloader_storage(UnifiedInMemoryStorage()) is False


class TestUnifiedAdapters:
    @pytest.mark.asyncio
    async def test_store_frames_and_returns_key_as_reference(self):
        storage = UnifiedInMemoryStorage()
        ref = await _store_content(storage, "tool_1_0", b"payload", "application/json")
        assert ref == "tool_1_0"
        content, content_type = await _retrieve_content(storage, ref)
        assert content == b"payload"
        assert content_type == "application/json"

    @pytest.mark.asyncio
    async def test_retrieve_missing_reference_raises_keyerror(self):
        with pytest.raises(KeyError):
            await _retrieve_content(UnifiedInMemoryStorage(), "does-not-exist")

    @pytest.mark.asyncio
    async def test_store_delegates_to_legacy_backend(self):
        storage = InMemoryStorage()
        ref = await _store_content(storage, "k", b"data", "text/plain")
        content, content_type = await _retrieve_content(storage, ref)
        assert content == b"data"
        assert content_type == "text/plain"


class TestPluginWithUnifiedStorage:
    @pytest.mark.asyncio
    async def test_offloads_text_block_to_one_framed_key(self):
        storage = UnifiedInMemoryStorage()
        plugin = ContextOffloader(
            storage=storage, max_result_tokens=25, preview_tokens=10, include_retrieval_tool=False
        )
        agent = _agent()

        await plugin._handle_tool_result(_after_tool_event(agent, [{"text": "x" * 400}]))

        assert await storage.list("") == ["tool_123_0"]
        content, content_type = await _retrieve_content(storage, "tool_123_0")
        assert content == b"x" * 400
        assert content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_offloads_json_block_preserving_content_type(self):
        storage = UnifiedInMemoryStorage()
        plugin = ContextOffloader(storage=storage, max_result_tokens=10, preview_tokens=5, include_retrieval_tool=False)
        agent = _agent()
        payload = {"data": "y" * 400}

        await plugin._handle_tool_result(_after_tool_event(agent, [{"json": payload}]))

        content, content_type = await _retrieve_content(storage, "tool_123_0")
        assert content_type == "application/json"
        assert json.loads(content) == payload

    @pytest.mark.asyncio
    async def test_retrieval_tool_reads_back_offloaded_content(self):
        storage = UnifiedInMemoryStorage()
        plugin = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10, include_retrieval_tool=True)
        agent = _agent()
        await plugin._handle_tool_result(_after_tool_event(agent, [{"text": "z" * 400}]))

        tool_context = MagicMock()
        tool_context.agent = agent
        result = await plugin.retrieve_offloaded_content("tool_123_0", tool_context=tool_context)

        assert result == "z" * 400


class TestUnifiedEviction:
    @pytest.mark.asyncio
    async def test_evicts_entries_older_than_window(self):
        storage = UnifiedInMemoryStorage()
        plugin = ContextOffloader(
            storage=storage,
            max_result_tokens=25,
            preview_tokens=10,
            include_retrieval_tool=False,
            evict_after_cycles=3,
        )
        agent = _agent(cycle_count=0)

        await plugin._handle_tool_result(_after_tool_event(agent, [{"text": "x" * 400}]))
        assert await storage.list("") == ["tool_123_0"]

        # cycle 3: threshold = 3 - 3 = 0, stored at 0, 0 < 0 is False -> retained
        await plugin._on_before_model_call(_before_model_event(agent, 3))
        assert await storage.list("") == ["tool_123_0"]

        # cycle 4: threshold = 4 - 3 = 1, stored at 0, 0 < 1 -> evicted
        await plugin._on_before_model_call(_before_model_event(agent, 4))
        assert await storage.list("") == []

    @pytest.mark.asyncio
    async def test_no_eviction_when_disabled(self):
        storage = UnifiedInMemoryStorage()
        plugin = ContextOffloader(
            storage=storage,
            max_result_tokens=25,
            preview_tokens=10,
            include_retrieval_tool=False,
            evict_after_cycles=None,
        )
        agent = _agent(cycle_count=0)

        await plugin._handle_tool_result(_after_tool_event(agent, [{"text": "x" * 400}]))
        await plugin._on_before_model_call(_before_model_event(agent, 10_000))

        assert await storage.list("") == ["tool_123_0"]


class TestEvictAfterCyclesValidation:
    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValueError, match="evict_after_cycles"):
            ContextOffloader(storage=UnifiedInMemoryStorage(), evict_after_cycles=bad)

    def test_accepts_none_and_positive(self):
        ContextOffloader(storage=UnifiedInMemoryStorage(), evict_after_cycles=None)
        ContextOffloader(storage=UnifiedInMemoryStorage(), evict_after_cycles=1)

    def test_rejects_bool(self):
        # bool is an int subclass; True must not be silently accepted as 1.
        with pytest.raises(ValueError, match="evict_after_cycles"):
            ContextOffloader(storage=UnifiedInMemoryStorage(), evict_after_cycles=True)


class TestLegacyWindowForwarding:
    def test_forwards_window_when_legacy_storage_at_default(self):
        storage = InMemoryStorage()  # default window (20)
        plugin = ContextOffloader(storage=storage, evict_after_cycles=5)

        plugin.init_agent(MagicMock())

        assert storage._evict_after_turns == 5

    def test_does_not_override_explicit_legacy_window(self):
        storage = InMemoryStorage(evict_after_turns=100)
        plugin = ContextOffloader(storage=storage, evict_after_cycles=5)

        plugin.init_agent(MagicMock())

        assert storage._evict_after_turns == 100
