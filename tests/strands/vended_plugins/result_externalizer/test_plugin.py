"""Tests for the ToolResultExternalizer plugin."""

import json
import logging
from unittest.mock import MagicMock

import pytest

from strands.hooks.events import AfterToolCallEvent
from strands.vended_plugins.result_externalizer import (
    InMemoryExternalizationStorage,
    ToolResultExternalizer,
)


@pytest.fixture
def storage():
    return InMemoryExternalizationStorage()


@pytest.fixture
def plugin(storage):
    return ToolResultExternalizer(
        storage=storage,
        max_result_chars=100,
        preview_chars=50,
    )


@pytest.fixture
def mock_agent():
    return MagicMock()


def _make_event(agent, text_content, status="success", tool_use_id="tool_123", cancel_message=None):
    """Helper to create an AfterToolCallEvent with text content."""
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


class TestToolResultExternalizer:
    def test_plugin_name(self, plugin):
        assert plugin.name == "result_externalizer"

    def test_hook_auto_discovered(self, plugin):
        assert len(plugin.hooks) == 1
        assert plugin.hooks[0].__name__ == "_handle_tool_result"

    def test_raises_on_non_positive_max_result_chars(self):
        with pytest.raises(ValueError, match="max_result_chars must be positive"):
            ToolResultExternalizer(storage=InMemoryExternalizationStorage(), max_result_chars=0)
        with pytest.raises(ValueError, match="max_result_chars must be positive"):
            ToolResultExternalizer(storage=InMemoryExternalizationStorage(), max_result_chars=-1)

    def test_raises_on_negative_preview_chars(self):
        with pytest.raises(ValueError, match="preview_chars must be non-negative"):
            ToolResultExternalizer(storage=InMemoryExternalizationStorage(), preview_chars=-1)

    def test_raises_on_preview_chars_gte_max_result_chars(self):
        with pytest.raises(ValueError, match="preview_chars must be less than max_result_chars"):
            ToolResultExternalizer(storage=InMemoryExternalizationStorage(), max_result_chars=100, preview_chars=100)
        with pytest.raises(ValueError, match="preview_chars must be less than max_result_chars"):
            ToolResultExternalizer(storage=InMemoryExternalizationStorage(), max_result_chars=100, preview_chars=200)

    def test_externalizes_oversized_text(self, plugin, storage, mock_agent):
        large_text = "a" * 200
        event = _make_event(mock_agent, large_text)

        plugin._handle_tool_result(event)

        # Result should be replaced with preview
        result_text = event.result["content"][0]["text"]
        assert "[Externalized: 200 chars | ref:" in result_text
        assert "[Full output stored externally:" in result_text
        # Preview should contain exactly preview_chars (50) of the original content
        assert "a" * 50 in result_text
        assert "a" * 51 not in result_text

        # Full content should be in storage
        ref = result_text.split("ref: ")[1].split("]")[0]
        assert storage.retrieve(ref) == large_text

    def test_preserves_status_and_tool_use_id(self, plugin, mock_agent):
        event = _make_event(mock_agent, "x" * 200, status="error", tool_use_id="my_tool_456")

        plugin._handle_tool_result(event)

        assert event.result["status"] == "error"
        assert event.result["toolUseId"] == "my_tool_456"

    def test_under_threshold_passes_through(self, plugin, mock_agent):
        small_text = "x" * 50
        event = _make_event(mock_agent, small_text)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_at_threshold_passes_through(self, plugin, mock_agent):
        exact_text = "x" * 100  # exactly at threshold
        event = _make_event(mock_agent, exact_text)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_skips_cancelled_tool_calls(self, plugin, mock_agent):
        large_text = "x" * 200
        event = _make_event(mock_agent, large_text, cancel_message="tool cancelled by user")
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_image_only_content_passes_through(self, plugin, mock_agent):
        content = [{"image": {"format": "png", "source": {"bytes": b"fake"}}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_mixed_content_replaces_images_with_placeholders(self, plugin, mock_agent):
        content = [
            {"text": "x" * 200},
            {"image": {"format": "png", "source": {"bytes": b"x" * 1024}}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        # Should have preview text block + image placeholder
        assert len(event.result["content"]) == 2
        assert "[Externalized:" in event.result["content"][0]["text"]
        assert event.result["content"][1]["text"] == "[image: png, 1024 bytes]"

    def test_document_blocks_replaced_with_placeholder(self, plugin, mock_agent):
        content = [
            {"text": "x" * 200},
            {"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": b"pdf" * 100}}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        assert len(event.result["content"]) == 2
        assert "[Externalized:" in event.result["content"][0]["text"]
        assert event.result["content"][1]["text"] == "[document: pdf, report.pdf, 300 bytes]"

    def test_multiple_text_blocks_concatenated(self, plugin, storage, mock_agent):
        content = [
            {"text": "a" * 60},
            {"text": "b" * 60},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        # Total is 120 chars > 100 threshold, should externalize
        result_text = event.result["content"][0]["text"]
        assert "[Externalized: 121 chars" in result_text  # 60 + \n + 60

        # Verify full content in storage
        ref = result_text.split("ref: ")[1].split("]")[0]
        stored = storage.retrieve(ref)
        assert stored == ("a" * 60 + "\n" + "b" * 60)

    def test_json_content_externalized(self, plugin, storage, mock_agent):
        large_json = {"data": [{"id": i, "value": "x" * 20} for i in range(10)]}
        content = [{"json": large_json}]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        result_text = event.result["content"][0]["text"]
        assert "[Externalized:" in result_text

        # Verify stored content is the JSON serialized with indent=2
        ref = result_text.split("ref: ")[1].split("]")[0]
        stored = storage.retrieve(ref)
        assert stored == json.dumps(large_json, indent=2)

    def test_mixed_text_and_json_concatenated(self, plugin, storage, mock_agent):
        content = [
            {"text": "a" * 60},
            {"json": {"key": "b" * 60}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        result_text = event.result["content"][0]["text"]
        assert "[Externalized:" in result_text

        # Both text and JSON should be in stored content
        ref = result_text.split("ref: ")[1].split("]")[0]
        stored = storage.retrieve(ref)
        assert "a" * 60 in stored
        assert "b" * 60 in stored

    def test_small_json_passes_through(self, plugin, mock_agent):
        content = [{"json": {"key": "value"}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_error_status_still_externalized(self, plugin, mock_agent):
        large_text = "x" * 200
        event = _make_event(mock_agent, large_text, status="error")

        plugin._handle_tool_result(event)

        assert "[Externalized:" in event.result["content"][0]["text"]
        assert event.result["status"] == "error"

    def test_storage_failure_keeps_original(self, mock_agent, caplog):
        failing_storage = MagicMock()
        failing_storage.store.side_effect = RuntimeError("disk full")

        plugin = ToolResultExternalizer(
            storage=failing_storage,
            max_result_chars=100,
            preview_chars=50,
        )

        large_text = "x" * 200
        event = _make_event(mock_agent, large_text)

        with caplog.at_level(logging.WARNING):
            plugin._handle_tool_result(event)

        # Original result should be unchanged
        assert event.result["content"][0]["text"] == large_text
        assert "failed to externalize" in caplog.text

    def test_empty_text_blocks_ignored_in_size_calculation(self, plugin, mock_agent):
        content = [
            {"text": ""},
            {"text": "x" * 50},
        ]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        # Total is 50 chars <= 100 threshold, should not externalize
        assert event.result["content"] is original_content

    def test_image_placeholder_format(self, plugin, mock_agent):
        content = [
            {"text": "x" * 200},
            {"image": {"format": "jpeg", "source": {"bytes": b"img" * 100}}},
            {"image": {"format": "png", "source": {}}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        assert event.result["content"][1]["text"] == "[image: jpeg, 300 bytes]"
        assert event.result["content"][2]["text"] == "[image: png, 0 bytes]"

    def test_document_only_content_passes_through(self, plugin, mock_agent):
        content = [{"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": b"pdf"}}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_document_placeholder_format(self, plugin, mock_agent):
        content = [
            {"text": "x" * 200},
            {"document": {"format": "csv", "name": "data.csv", "source": {"bytes": b"a,b\n1,2" * 50}}},
            {"document": {"format": "unknown", "name": "unknown", "source": {}}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        assert event.result["content"][1]["text"] == "[document: csv, data.csv, 350 bytes]"
        assert event.result["content"][2]["text"] == "[document: unknown, unknown, 0 bytes]"

    def test_all_content_types_mixed(self, plugin, storage, mock_agent):
        large_json = {"rows": [{"id": i} for i in range(20)]}
        content = [
            {"text": "a" * 60},
            {"json": large_json},
            {"image": {"format": "png", "source": {"bytes": b"img" * 500}}},
            {"document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": b"pdf" * 200}}},
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        result_content = event.result["content"]
        # Preview + image placeholder + document placeholder = 3 blocks
        assert len(result_content) == 3
        assert "[Externalized:" in result_content[0]["text"]
        assert result_content[1]["text"] == "[image: png, 1500 bytes]"
        assert result_content[2]["text"] == "[document: pdf, report.pdf, 600 bytes]"

        # Verify stored content has both text and JSON
        ref = result_content[0]["text"].split("ref: ")[1].split("]")[0]
        stored = storage.retrieve(ref)
        assert "a" * 60 in stored
        assert json.dumps(large_json, indent=2) in stored
