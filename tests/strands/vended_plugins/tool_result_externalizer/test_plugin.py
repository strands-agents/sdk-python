"""Tests for the ToolResultExternalizer plugin."""

import logging
from unittest.mock import MagicMock

import pytest

from strands.hooks.events import AfterToolCallEvent
from strands.vended_plugins.tool_result_externalizer import (
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
        size_threshold_chars=100,
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
        assert plugin.name == "tool_result_externalizer"

    def test_hook_auto_discovered(self, plugin):
        assert len(plugin.hooks) == 1
        assert plugin.hooks[0].__name__ == "_handle_tool_result"

    def test_default_storage_is_in_memory(self):
        plugin = ToolResultExternalizer()
        assert isinstance(plugin._storage, InMemoryExternalizationStorage)

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

    def test_non_text_content_passes_through(self, plugin, mock_agent):
        content = [{"image": {"format": "png", "source": {"bytes": b"fake"}}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content

    def test_mixed_content_preserves_non_text(self, plugin, mock_agent):
        image_block = {"image": {"format": "png", "source": {"bytes": b"fake"}}}
        content = [
            {"text": "x" * 200},
            image_block,
        ]
        event = _make_event(mock_agent, content)

        plugin._handle_tool_result(event)

        # Should have preview text block + original image block
        assert len(event.result["content"]) == 2
        assert "text" in event.result["content"][0]
        assert "[Externalized:" in event.result["content"][0]["text"]
        assert event.result["content"][1] is image_block

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
            size_threshold_chars=100,
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

    def test_json_only_content_passes_through(self, plugin, mock_agent):
        content = [{"json": {"key": "value"}}]
        event = _make_event(mock_agent, content)
        original_content = event.result["content"]

        plugin._handle_tool_result(event)

        assert event.result["content"] is original_content
