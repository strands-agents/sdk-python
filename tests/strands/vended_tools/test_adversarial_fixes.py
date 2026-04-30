"""Adversarial tests validating the bug fixes.

These tests verify that all 6 findings from the adversarial testing report
are now resolved.
"""

import pytest

from strands.sandbox.base import ExecutionResult, StreamChunk
from strands.vended_tools.shell import shell, _CWD_MARKER
from strands.vended_tools.editor import editor, _UNDO_STATE_KEY, STATE_KEY

from .conftest import collect_generator


class TestShellCwdMarkerInjectionFixed:
    """Verify Fix #1: CWD marker injection no longer corrupts state."""

    @pytest.mark.asyncio
    async def test_user_output_marker_does_not_corrupt_cwd(self, tool_context, tmp_path):
        """User echoing the marker string should not affect tracked cwd."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command=f"echo '{_CWD_MARKER}' && echo '/evil/path'",
                tool_context=tool_context,
            )
        )

        shell_state = tool_context.agent.state.get("_strands_shell_state") or {}
        tracked_cwd = shell_state.get("cwd", "")

        # CWD should NOT be /evil/path or contain the marker
        assert tracked_cwd != "/evil/path"
        assert _CWD_MARKER not in tracked_cwd
        # It SHOULD be a real, valid single-line path
        assert "\n" not in tracked_cwd
        assert tracked_cwd != ""

    @pytest.mark.asyncio
    async def test_multiple_marker_outputs_still_track_correctly(self, tool_context, tmp_path):
        """Multiple user-output markers should not corrupt state."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command=f"echo '{_CWD_MARKER}' && echo '{_CWD_MARKER}' && echo fake_path",
                tool_context=tool_context,
            )
        )

        shell_state = tool_context.agent.state.get("_strands_shell_state") or {}
        tracked_cwd = shell_state.get("cwd", "")

        assert tracked_cwd != "fake_path"
        assert _CWD_MARKER not in tracked_cwd
        assert "\n" not in tracked_cwd
        assert tracked_cwd != ""

    @pytest.mark.asyncio
    async def test_subsequent_calls_work_after_marker_injection(self, tool_context, tmp_path):
        """After a marker injection attempt, next shell call should still work."""
        # First call: inject marker
        await collect_generator(
            shell.__wrapped__(
                command=f"echo '{_CWD_MARKER}' && echo '/evil/path'",
                tool_context=tool_context,
            )
        )

        # Second call: should work normally (no Permission denied)
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo follow_up", tool_context=tool_context)
        )
        assert "follow_up" in result
        assert "error" not in result.lower()


class TestShellStringTimeoutFixed:
    """Verify Fix #3: String timeout no longer crashes."""

    @pytest.mark.asyncio
    async def test_string_timeout_coerced_to_int(self, tool_context, mock_agent):
        """String timeout "30" should be coerced to int 30."""
        mock_agent.state.set("strands_shell_tool", {"timeout": "30"})

        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo test", tool_context=tool_context)
        )
        # Should not crash — timeout is coerced
        assert "test" in result
        assert "error" not in result.lower()

    @pytest.mark.asyncio
    async def test_float_timeout_coerced(self, tool_context, mock_agent):
        """Float timeout 30.5 should be coerced to int 30."""
        mock_agent.state.set("strands_shell_tool", {"timeout": 30.5})

        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo test", tool_context=tool_context)
        )
        assert "test" in result

    @pytest.mark.asyncio
    async def test_invalid_timeout_uses_default(self, tool_context, mock_agent):
        """Non-numeric timeout should fall back to default."""
        mock_agent.state.set("strands_shell_tool", {"timeout": "not_a_number"})

        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo test", tool_context=tool_context)
        )
        assert "test" in result


class TestShellMarkerSplitFixed:
    """Verify Fix #5: Marker split across chunks no longer leaks."""

    @pytest.mark.asyncio
    async def test_marker_split_across_chunks_no_leak(self, tool_context, mock_agent):
        """Marker split across chunks should not leak partial marker to consumer."""
        from unittest.mock import AsyncMock

        async def fake_streaming(command, timeout=None, cwd=None):
            yield StreamChunk(data="output\n__STRAN", stream_type="stdout")
            yield StreamChunk(data="DS_CWD__\n/tmp\n", stream_type="stdout")
            yield ExecutionResult(
                exit_code=0,
                stdout="output\n__STRANDS_CWD__\n/tmp\n",
                stderr="",
            )

        mock_agent.sandbox = AsyncMock()
        mock_agent.sandbox.execute_streaming = fake_streaming

        chunks, result = await collect_generator(
            shell.__wrapped__(command="test", tool_context=tool_context)
        )

        # No partial marker should leak
        all_chunk_data = "".join(c.data for c in chunks if c.stream_type == "stdout")
        assert "__STRAN" not in all_chunk_data
        assert _CWD_MARKER not in all_chunk_data
        # User output should be preserved
        assert "output" in all_chunk_data


class TestEditorTabExpansionFixed:
    """Verify Fixes #2 and #4: Tab expansion no longer corrupts files or creates false matches."""

    @pytest.mark.asyncio
    async def test_tabs_preserved_after_edit(self, tool_context, sandbox, tmp_path):
        """Editing a file should NOT destroy tab characters."""
        path = f"{tmp_path}/preserve_tabs.txt"
        original = "def foo():\n\treturn 42\n"
        await sandbox.write_text(path, original)

        # Edit something unrelated to tabs
        result = await editor.__wrapped__(
            command="str_replace",
            path=path,
            old_str="42",
            new_str="99",
            tool_context=tool_context,
        )

        content = await sandbox.read_text(path)
        # Tab MUST still be present
        assert "\t" in content, f"Tab was destroyed! Content: {repr(content)}"
        assert content == "def foo():\n\treturn 99\n"

    @pytest.mark.asyncio
    async def test_tab_and_spaces_not_conflated(self, tool_context, sandbox, tmp_path):
        """Tab and 8 spaces should NOT be treated as the same thing."""
        path = f"{tmp_path}/tabs.txt"
        content = "\thello\n        hello\n"
        await sandbox.write_text(path, content)

        # Replace the tab version — should work (unique in source)
        result = await editor.__wrapped__(
            command="str_replace",
            path=path,
            old_str="\thello",
            new_str="replaced",
            tool_context=tool_context,
        )

        # Should succeed — no "multiple occurrences" error
        assert "edited" in result.lower(), f"Failed: {result}"
        new_content = await sandbox.read_text(path)
        assert "replaced" in new_content
        assert "        hello" in new_content  # 8-space version unchanged

    @pytest.mark.asyncio
    async def test_insert_preserves_tabs(self, tool_context, sandbox, tmp_path):
        """Insert should not expand tabs in existing content."""
        path = f"{tmp_path}/insert_tabs.txt"
        await sandbox.write_text(path, "line1\n\tindented\nline3")

        result = await editor.__wrapped__(
            command="insert",
            path=path,
            insert_line=1,
            new_str="new_line",
            tool_context=tool_context,
        )

        content = await sandbox.read_text(path)
        assert "\t" in content, f"Tab destroyed by insert! Content: {repr(content)}"


class TestEditorFloatViewRangeFixed:
    """Verify Fix #6: Float view_range no longer crashes."""

    @pytest.mark.asyncio
    async def test_float_view_range_coerced(self, tool_context, sandbox, tmp_path):
        """view_range with floats [1.0, 3.0] should work (coerced to ints)."""
        path = f"{tmp_path}/float_range.txt"
        await sandbox.write_text(path, "line1\nline2\nline3\nline4\nline5")

        result = await editor.__wrapped__(
            command="view",
            path=path,
            view_range=[1.0, 3.0],
            tool_context=tool_context,
        )

        # Should work, showing lines 1-3
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "line4" not in result

    @pytest.mark.asyncio
    async def test_invalid_view_range_type_gives_error(self, tool_context, sandbox, tmp_path):
        """Non-numeric view_range should give a clear error."""
        path = f"{tmp_path}/bad_range.txt"
        await sandbox.write_text(path, "line1\nline2\nline3")

        result = await editor.__wrapped__(
            command="view",
            path=path,
            view_range=["a", "b"],
            tool_context=tool_context,
        )

        assert "error" in result.lower()
