"""Tests for the ContextOffloader plugin."""

import json
import logging
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from strands.hooks.events import AfterToolCallEvent
from strands.types.tools import ToolContext, ToolUse
from strands.vended_plugins.context_offloader import (
    ContextOffloader,
    InMemoryStorage,
)


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def plugin(storage):
    return ContextOffloader(
        storage=storage,
        max_result_tokens=25,
        preview_tokens=10,
    )


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.model = MagicMock()
    agent.model.count_tokens = AsyncMock(side_effect=_heuristic_count_tokens)
    return agent


async def _heuristic_count_tokens(messages, **kwargs):
    """Heuristic token counter for tests: chars / 4."""
    total = 0
    for msg in messages:
        for block in msg.get("content", []):
            if "toolResult" in block:
                for content in block["toolResult"].get("content", []):
                    if "text" in content:
                        total += math.ceil(len(content["text"]) / 4)
                    elif "json" in content:
                        total += math.ceil(len(json.dumps(content["json"])) / 4)
            elif "text" in block:
                total += math.ceil(len(block["text"]) / 4)
    return total


def _make_event(agent, text_content, status="success", tool_use_id="tool_123", cancel_message=None):
    """Helper to create an AfterToolCallEvent with content."""
    if isinstance(text_content, str):
        content = [{"text": text_content}]
    else:
        content = text_content

    result = {
        "toolUseId": tool_use_id,
        "status": status,
        "content": content,
    }
    tool_use = {"toolUseId": tool_use_id, "name": "test_tool", "input": {}}

    return AfterToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use=tool_use,
        invocation_state={},
        result=result,
        cancel_message=cancel_message,
    )


class TestContextOffloader:
    def test_plugin_name(self, plugin):
        assert plugin.name == "context_offloader"

    def test_hooks_auto_discovered(self, plugin):
        assert len(plugin.hooks) == 1
        assert plugin.hooks[0].__name__ == "_handle_tool_result"

    def test_raises_on_non_positive_max_result_tokens(self):
        with pytest.raises(ValueError, match="max_result_tokens must be positive"):
            ContextOffloader(storage=InMemoryStorage(), max_result_tokens=0)
        with pytest.raises(ValueError, match="max_result_tokens must be positive"):
            ContextOffloader(storage=InMemoryStorage(), max_result_tokens=-1)

    def test_raises_on_negative_preview_tokens(self):
        with pytest.raises(ValueError, match="preview_tokens must be non-negative"):
            ContextOffloader(storage=InMemoryStorage(), preview_tokens=-1)

    def test_raises_on_preview_tokens_gte_max_result_tokens(self):
        with pytest.raises(ValueError, match="preview_tokens must be less than max_result_tokens"):
            ContextOffloader(storage=InMemoryStorage(), max_result_tokens=100, preview_tokens=100)
        with pytest.raises(ValueError, match="preview_tokens must be less than max_result_tokens"):
            ContextOffloader(storage=InMemoryStorage(), max_result_tokens=100, preview_tokens=200)

    @pytest.mark.asyncio
    async def test_offloads_oversized_text(self, plugin, storage, mock_agent):
        large_text = "a" * 200
        event = _make_event(mock_agent, large_text)

        await plugin._handle_tool_result(event)

        result_text = event.result["content"][0]["text"]
        assert "[Offloaded:" in result_text
        # Preview should be shorter than the full text
        assert len(result_text) < len(large_text) + 500  # preview + metadata < original + overhead

        # Verify stored content
        assert len(storage._store) == 1
        ref = list(storage._store.keys())[0]
        content, content_type = storage.retrieve(ref)
        assert content == large_text.encode("utf-8")
        assert content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_preserves_status_and_tool_use_id(self, plugin, mock_agent):
        event = _make_event(mock_agent, "x" * 200, status="error", tool_use_id="my_tool_456")

        await plugin._handle_tool_result(event)

        assert event.result["status"] == "error"
        assert event.result["toolUseId"] == "my_tool_456"

    @pytest.mark.asyncio
    async def test_under_threshold_passes_through(self, plugin, mock_agent):
        small_text = "x" * 50  # 12.5 tokens, under 25
        event = _make_event(mock_agent, small_text)
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_at_threshold_passes_through(self, plugin, mock_agent):
        exact_text = "x" * 100  # exactly 25 tokens
        event = _make_event(mock_agent, exact_text)
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_skips_cancelled_tool_calls(self, plugin, mock_agent):
        large_text = "x" * 200
        event = _make_event(mock_agent, large_text, cancel_message="tool cancelled by user")
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_skips_retrieve_tool_results_when_enabled(self, storage, mock_agent):
        plugin = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10, include_retrieval_tool=True)
        large_text = "x" * 200
        result = {"toolUseId": "tool_123", "status": "success", "content": [{"text": large_text}]}
        tool_use = {"toolUseId": "tool_123", "name": plugin.retrieve_offloaded_content.tool_name, "input": {}}
        event = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use=tool_use,
            invocation_state={},
            result=result,
        )
        await plugin._handle_tool_result(event)

        assert event.result["content"][0]["text"] == large_text

    @pytest.mark.asyncio
    async def test_does_not_skip_retrieve_tool_when_disabled(self, plugin, storage, mock_agent):
        large_text = "x" * 200
        result = {"toolUseId": "tool_123", "status": "success", "content": [{"text": large_text}]}
        tool_use = {"toolUseId": "tool_123", "name": "retrieve_offloaded_content", "input": {}}
        event = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use=tool_use,
            invocation_state={},
            result=result,
        )
        await plugin._handle_tool_result(event)

        # Tool is disabled, so the result should be offloaded normally
        assert "[Offloaded:" in event.result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_image_only_content_passes_through(self, plugin, mock_agent):
        content = [{"image": {"format": "png", "source": {"bytes": b"fake"}}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_image_stored_and_placeholder_has_ref(self, plugin, storage, mock_agent):
        img_bytes = b"\x89PNG" + b"\x00" * 100
        content = [
            {"text": "x" * 200},
            {"image": {"format": "png", "source": {"bytes": img_bytes}}},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Should have preview + image placeholder
        assert len(event.result["content"]) == 2
        placeholder = event.result["content"][1]["text"]
        assert "[image: png, 104 bytes" in placeholder
        assert "ref:" in placeholder

        # Verify image was stored
        assert len(storage._store) == 2  # text + image
        img_ref = placeholder.split("ref: ")[1].rstrip("]")
        img_content, img_type = storage.retrieve(img_ref)
        assert img_content == img_bytes
        assert img_type == "image/png"

    @pytest.mark.asyncio
    async def test_document_stored_and_placeholder_has_ref(self, plugin, storage, mock_agent):
        doc_bytes = b"%PDF-1.4" + b"\x00" * 100
        content = [
            {"text": "x" * 200},
            {"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": doc_bytes}}},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        assert len(event.result["content"]) == 2
        placeholder = event.result["content"][1]["text"]
        assert "[document: pdf, report.pdf, 108 bytes" in placeholder
        assert "ref:" in placeholder

        # Verify document was stored
        doc_ref = placeholder.split("ref: ")[1].rstrip("]")
        doc_content, doc_type = storage.retrieve(doc_ref)
        assert doc_content == doc_bytes
        assert doc_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_stored_separately(self, plugin, storage, mock_agent):
        content = [
            {"text": "a" * 60},
            {"text": "b" * 60},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Two text blocks stored separately
        assert len(storage._store) == 2
        refs = list(storage._store.keys())
        assert storage.retrieve(refs[0]) == (b"a" * 60, "text/plain")
        assert storage.retrieve(refs[1]) == (b"b" * 60, "text/plain")

    @pytest.mark.asyncio
    async def test_json_content_stored_as_json(self, plugin, storage, mock_agent):
        large_json = {"data": [{"id": i, "value": "x" * 20} for i in range(10)]}
        content = [{"json": large_json}]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        assert len(storage._store) == 1
        ref = list(storage._store.keys())[0]
        stored_content, content_type = storage.retrieve(ref)
        assert content_type == "application/json"
        assert json.loads(stored_content) == large_json

    @pytest.mark.asyncio
    async def test_mixed_text_and_json(self, plugin, storage, mock_agent):
        content = [
            {"text": "a" * 60},
            {"json": {"key": "b" * 60}},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Both stored separately with correct types
        assert len(storage._store) == 2
        refs = list(storage._store.keys())
        assert storage.retrieve(refs[0])[1] == "text/plain"
        assert storage.retrieve(refs[1])[1] == "application/json"

    @pytest.mark.asyncio
    async def test_small_json_passes_through(self, plugin, mock_agent):
        content = [{"json": {"key": "value"}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_error_status_still_offloaded(self, plugin, mock_agent):
        large_text = "x" * 200
        event = _make_event(mock_agent, large_text, status="error")

        await plugin._handle_tool_result(event)

        assert "[Offloaded:" in event.result["content"][0]["text"]
        assert event.result["status"] == "error"

    @pytest.mark.asyncio
    async def test_storage_failure_keeps_original(self, mock_agent, caplog):
        failing_storage = MagicMock()
        failing_storage.store.side_effect = RuntimeError("disk full")

        plugin = ContextOffloader(
            storage=failing_storage,
            max_result_tokens=25,
            preview_tokens=10,
        )

        large_text = "x" * 200
        event = _make_event(mock_agent, large_text)

        with caplog.at_level(logging.WARNING):
            await plugin._handle_tool_result(event)

        assert event.result["content"][0]["text"] == large_text
        assert "failed to offload" in caplog.text

    @pytest.mark.asyncio
    async def test_partial_storage_failure_keeps_original(self, mock_agent, caplog):
        storage = MagicMock()
        call_count = 0

        def store_then_fail(key, content, content_type="text/plain"):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("disk full on second block")
            return f"ref_{call_count}"

        storage.store.side_effect = store_then_fail

        plugin = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10)

        content = [
            {"text": "a" * 60},
            {"text": "b" * 60},
        ]
        event = _make_event(mock_agent, content)

        with caplog.at_level(logging.WARNING):
            await plugin._handle_tool_result(event)

        assert event.result["content"][0]["text"] == "a" * 60
        assert event.result["content"][1]["text"] == "b" * 60
        assert "failed to offload" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_text_blocks_not_stored(self, plugin, storage, mock_agent):
        content = [
            {"text": ""},
            {"text": "x" * 200},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Empty text block is not in text_preview_parts but still iterated for storage
        # The non-empty block triggers offloading
        assert "[Offloaded:" in event.result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_document_only_content_passes_through(self, plugin, mock_agent):
        content = [{"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": b"pdf"}}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        await plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    @pytest.mark.asyncio
    async def test_unknown_content_type_passed_through(self, plugin, mock_agent):
        unknown_block = {"custom_type": {"data": "something"}}
        content = [
            {"text": "x" * 200},
            unknown_block,
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Unknown block should be passed through
        assert event.result["content"][-1] is unknown_block

    @pytest.mark.asyncio
    async def test_all_content_types_mixed(self, plugin, storage, mock_agent):
        large_json = {"rows": [{"id": i} for i in range(20)]}
        img_bytes = b"\x89PNG" + b"\x00" * 100
        doc_bytes = b"%PDF" + b"\x00" * 200
        content = [
            {"text": "a" * 60},
            {"json": large_json},
            {"image": {"format": "png", "source": {"bytes": img_bytes}}},
            {"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": doc_bytes}}},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        result_content = event.result["content"]
        # Preview + image placeholder + document placeholder = 3 blocks
        assert len(result_content) == 3
        assert "[Offloaded:" in result_content[0]["text"]
        assert "[image: png" in result_content[1]["text"]
        assert "[document: pdf, report.pdf" in result_content[2]["text"]

        # All 4 blocks stored
        assert len(storage._store) == 4

    @pytest.mark.asyncio
    async def test_image_without_bytes_not_stored(self, plugin, storage, mock_agent):
        content = [
            {"text": "x" * 200},
            {"image": {"format": "png", "source": {}}},
        ]
        event = _make_event(mock_agent, content)

        await plugin._handle_tool_result(event)

        # Only text stored, not the empty image
        assert len(storage._store) == 1
        placeholder = event.result["content"][1]["text"]
        assert "0 bytes" in placeholder
        assert "ref:" not in placeholder


class TestRetrievalTool:
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def plugin(self, storage):
        return ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10, include_retrieval_tool=True)

    @pytest.fixture
    def mock_agent(self):
        return MagicMock()

    @pytest.fixture
    def tool_context(self, mock_agent):
        tool_use = ToolUse(toolUseId="retrieve_1", name="retrieve_offloaded_content", input={})
        return ToolContext(tool_use=tool_use, agent=mock_agent, invocation_state={})

    def test_retrieval_tool_registered_when_enabled(self, plugin):
        tool_names = [t.tool_name for t in plugin.tools]
        assert "retrieve_offloaded_content" in tool_names

    def test_retrieval_tool_not_registered_by_default(self):
        plugin = ContextOffloader(storage=InMemoryStorage())
        plugin.init_agent(MagicMock())
        tool_names = [t.tool_name for t in plugin.tools]
        assert "retrieve_offloaded_content" not in tool_names

    def test_retrieve_text_content(self, plugin, storage, tool_context):
        ref = storage.store("key_1", b"hello world", "text/plain")
        result = plugin.retrieve_offloaded_content(reference=ref, tool_context=tool_context)
        assert result == "hello world"

    def test_retrieve_json_content(self, plugin, storage, tool_context):
        ref = storage.store("key_1", b'{"key": "value"}', "application/json")
        result = plugin.retrieve_offloaded_content(reference=ref, tool_context=tool_context)
        assert result["content"][0]["json"] == {"key": "value"}

    def test_retrieve_large_text_returns_full_content(self, plugin, storage, tool_context):
        large_text = "a" * 50_000
        ref = storage.store("key_1", large_text.encode("utf-8"), "text/plain")
        result = plugin.retrieve_offloaded_content(reference=ref, tool_context=tool_context)
        assert result == large_text

    def test_retrieve_missing_reference(self, plugin, tool_context):
        result = plugin.retrieve_offloaded_content(reference="nonexistent", tool_context=tool_context)
        assert "Error: reference not found" in result

    def test_retrieve_image_content(self, plugin, storage, tool_context):
        img_bytes = b"\x89PNG\x00\x00"
        ref = storage.store("key_1", img_bytes, "image/png")
        result = plugin.retrieve_offloaded_content(reference=ref, tool_context=tool_context)
        assert result["status"] == "success"
        assert result["content"][0]["image"]["format"] == "png"
        assert result["content"][0]["image"]["source"]["bytes"] == img_bytes

    def test_retrieve_document_content(self, plugin, storage, tool_context):
        doc_bytes = b"%PDF-1.4 content"
        ref = storage.store("key_1", doc_bytes, "application/pdf")
        result = plugin.retrieve_offloaded_content(reference=ref, tool_context=tool_context)
        assert result["status"] == "success"
        assert result["content"][0]["document"]["format"] == "pdf"
        assert result["content"][0]["document"]["source"]["bytes"] == doc_bytes


class TestInlineGuidance:
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.model = MagicMock()
        agent.model.count_tokens = AsyncMock(side_effect=_heuristic_count_tokens)
        return agent

    @pytest.mark.asyncio
    async def test_guidance_mentions_retrieval_tool_when_enabled(self, storage, mock_agent):
        plugin = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10, include_retrieval_tool=True)
        event = _make_event(mock_agent, "x" * 200)
        await plugin._handle_tool_result(event)
        result_text = event.result["content"][0]["text"]
        assert "retrieve_offloaded_content" in result_text

    @pytest.mark.asyncio
    async def test_guidance_does_not_mention_retrieval_tool_when_disabled(self, storage, mock_agent):
        plugin = ContextOffloader(storage=storage, max_result_tokens=25, preview_tokens=10)
        event = _make_event(mock_agent, "x" * 200)
        await plugin._handle_tool_result(event)
        result_text = event.result["content"][0]["text"]
        assert "retrieve_offloaded_content" not in result_text
        assert "available tools" in result_text
