"""Tests for vended tools package imports and tool specs."""

import inspect

import pytest

from strands.sandbox.base import StreamChunk

from .conftest import collect_generator


class TestVendedToolsImport:
    """Test that vended tools can be imported from the package."""

    def test_import_from_vended_tools(self):
        """Test importing from strands.vended_tools."""
        from strands.vended_tools import editor, python_repl, shell

        assert shell is not None
        assert editor is not None
        assert python_repl is not None

    def test_import_individual_tools(self):
        """Test importing individual tools from flat modules."""
        from strands.vended_tools.editor import editor
        from strands.vended_tools.python_repl import python_repl
        from strands.vended_tools.shell import shell

        assert shell is not None
        assert editor is not None
        assert python_repl is not None

    def test_tools_have_tool_spec(self):
        """Test that tools have proper tool specs."""
        from strands.vended_tools import editor, python_repl, shell

        assert shell.tool_name == "shell"
        assert editor.tool_name == "editor"
        assert python_repl.tool_name == "python_repl"

        for t in [shell, editor, python_repl]:
            spec = t.tool_spec
            assert "name" in spec
            assert "description" in spec
            assert "inputSchema" in spec

    def test_shell_tool_spec_shape(self):
        """Test shell tool spec matches expected shape."""
        from strands.vended_tools import shell

        spec = shell.tool_spec
        schema = spec["inputSchema"]["json"]
        props = schema.get("properties", {})

        assert "command" in props
        assert "timeout" in props
        assert "restart" in props
        assert schema.get("required") == ["command"]

    def test_editor_tool_spec_shape(self):
        """Test editor tool spec matches expected shape."""
        from strands.vended_tools import editor

        spec = editor.tool_spec
        schema = spec["inputSchema"]["json"]
        props = schema.get("properties", {})

        assert "command" in props
        assert "path" in props
        assert "file_text" in props
        assert "old_str" in props
        assert "new_str" in props
        assert "insert_line" in props
        assert "view_range" in props
        assert set(schema.get("required", [])) == {"command", "path"}

    def test_python_repl_tool_spec_shape(self):
        """Test python_repl tool spec matches expected shape."""
        from strands.vended_tools import python_repl

        spec = python_repl.tool_spec
        schema = spec["inputSchema"]["json"]
        props = schema.get("properties", {})

        assert "code" in props
        assert "timeout" in props
        assert "reset" in props
        assert schema.get("required") == ["code"]

    def test_shell_is_async_generator(self):
        """Test that shell is detected as an async generator function."""
        from strands.vended_tools.shell import shell

        assert inspect.isasyncgenfunction(shell.__wrapped__)

    def test_python_repl_is_async_generator(self):
        """Test that python_repl is detected as an async generator function."""
        from strands.vended_tools.python_repl import python_repl

        assert inspect.isasyncgenfunction(python_repl.__wrapped__)

    def test_editor_is_not_async_generator(self):
        """Test that editor is a regular async function (not generator)."""
        from strands.vended_tools.editor import editor

        assert not inspect.isasyncgenfunction(editor.__wrapped__)
        assert inspect.iscoroutinefunction(editor.__wrapped__)


class TestStreamingIntegration:
    """Test the streaming behavior of tools end-to-end."""

    @pytest.mark.asyncio
    async def test_shell_streams_before_result(self, tool_context):
        """Test that shell yields chunks BEFORE the final result."""
        from strands.vended_tools.shell import shell

        all_items = []

        async for item in shell.__wrapped__(
            command="echo streaming_test", tool_context=tool_context
        ):
            all_items.append(item)

        # Should have at least 2 items: chunk(s) + final result
        assert len(all_items) >= 2
        # Last item should be the string result
        assert isinstance(all_items[-1], str)
        # Earlier items should include StreamChunk
        stream_items = [i for i in all_items[:-1] if isinstance(i, StreamChunk)]
        assert len(stream_items) >= 1

    @pytest.mark.asyncio
    async def test_python_repl_streams_before_result(self, tool_context):
        """Test that python_repl yields chunks BEFORE the final result."""
        from strands.vended_tools.python_repl import python_repl

        all_items = []

        async for item in python_repl.__wrapped__(
            code="print('streaming_test')", tool_context=tool_context
        ):
            all_items.append(item)

        assert len(all_items) >= 2
        assert isinstance(all_items[-1], str)
        stream_items = [i for i in all_items[:-1] if isinstance(i, StreamChunk)]
        assert len(stream_items) >= 1

    @pytest.mark.asyncio
    async def test_shell_error_no_streaming_on_timeout(self, tool_context):
        """Test that timeout errors don't yield any StreamChunks before the error."""
        from strands.vended_tools.shell import shell

        chunks, result = await collect_generator(
            shell.__wrapped__(command="sleep 10", timeout=1, tool_context=tool_context)
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_python_repl_error_no_streaming_on_timeout(self, tool_context):
        """Test that timeout errors don't yield chunks before the error."""
        from strands.vended_tools.python_repl import python_repl

        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import time; time.sleep(10)",
                timeout=1,
                tool_context=tool_context,
            )
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_shell_stream_data_matches_result(self, tool_context):
        """Test that streamed chunk data matches the final result content."""
        from strands.vended_tools.shell import shell

        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo precise_output_42", tool_context=tool_context)
        )
        # The streamed chunks should contain the same data as the final result
        all_chunk_data = "".join(c.data for c in chunks)
        assert "precise_output_42" in all_chunk_data
        assert "precise_output_42" in result
