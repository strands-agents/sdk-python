"""Tests for the truncate method module."""

import pytest

from strands._context_manager.methods.truncate import (
    CHARS_PER_TOKEN,
    DEFAULT_PREVIEW_TOKENS,
    build_preview,
    truncate_text_block,
    truncate_tool_result,
)
from strands.types.content import ContentBlock
from strands.types.tools import ToolResult


class TestBuildPreview:
    """Tests for the build_preview function."""

    def test_short_text_returns_original(self):
        text = "short"
        tru_result = build_preview(text, 1, {})
        assert tru_result == text

    def test_head_mode(self):
        text = "a" * 200
        config = {"preview_tokens": 10, "preview": "head"}
        tru_result = build_preview(text, 1, config)
        assert "[Truncated:" in tru_result
        assert "a" * 10 in tru_result

    def test_tail_mode(self):
        text = "x" * 100 + "y" * 100
        config = {"preview_tokens": 10, "preview": "tail"}
        tru_result = build_preview(text, 1, config)
        assert "[Truncated:" in tru_result
        assert "y" * (10 * CHARS_PER_TOKEN) in tru_result

    def test_head_tail_mode_default(self):
        text = "a" * 200
        config = {"preview_tokens": 10}
        tru_result = build_preview(text, 1, config)
        assert "[Truncated:" in tru_result
        assert "a" in tru_result

    def test_uses_default_preview_tokens(self):
        text = "a" * (DEFAULT_PREVIEW_TOKENS * CHARS_PER_TOKEN * 3)
        tru_result = build_preview(text, 1, {})
        assert "[Truncated:" in tru_result


class TestTruncateToolResult:
    """Tests for truncate_tool_result."""

    def test_truncates_large_text_content(self):
        large_text = "x" * 50000
        tool_result = ToolResult(
            toolUseId="tool-1",
            status="success",
            content=[{"text": large_text}],
        )
        tru_result = truncate_tool_result(tool_result, {"preview_tokens": 100})
        text_content = tru_result["content"][0]["text"]
        assert "[Truncated:" in text_content
        assert len(text_content) < len(large_text)

    def test_preserves_short_content(self):
        tool_result = ToolResult(
            toolUseId="tool-1",
            status="success",
            content=[{"text": "short result"}],
        )
        tru_result = truncate_tool_result(tool_result, {"preview_tokens": 100})
        assert tru_result["content"][0]["text"] == "short result"

    def test_preserves_tool_use_id_and_status(self):
        tool_result = ToolResult(
            toolUseId="tool-abc",
            status="error",
            content=[{"text": "x" * 50000}],
        )
        tru_result = truncate_tool_result(tool_result, {"preview_tokens": 100})
        assert tru_result["toolUseId"] == "tool-abc"
        assert tru_result["status"] == "error"


class TestTruncateTextBlock:
    """Tests for truncate_text_block."""

    def test_truncates_large_text_block(self):
        block = ContentBlock(text="z" * 50000)
        tru_result = truncate_text_block(block, {"preview_tokens": 100})
        assert tru_result is not None
        assert "[Truncated:" in tru_result["text"]
        assert len(tru_result["text"]) < 50000

    def test_returns_original_for_short_text(self):
        block = ContentBlock(text="short")
        tru_result = truncate_text_block(block, {"preview_tokens": 100})
        assert tru_result is block

    @pytest.mark.parametrize("preview_mode", ["head", "tail", "head_tail"])
    def test_supports_all_preview_modes(self, preview_mode):
        block = ContentBlock(text="a" * 50000)
        tru_result = truncate_text_block(block, {"preview_tokens": 100, "preview": preview_mode})
        assert tru_result is not None
        assert "[Truncated:" in tru_result["text"]
