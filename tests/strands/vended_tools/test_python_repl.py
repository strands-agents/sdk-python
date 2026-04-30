"""Tests for the python_repl vended tool."""

import pytest

from strands.sandbox.base import StreamChunk
from strands.sandbox.noop import NoOpSandbox
from strands.vended_tools.python_repl import python_repl

from .conftest import collect_generator


class TestPythonReplTool:
    """Tests for the python_repl vended tool."""

    @pytest.mark.asyncio
    async def test_basic_code(self, tool_context):
        """Test basic Python code execution returns result."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="print('hello from python')",
                tool_context=tool_context,
            )
        )
        assert "hello from python" in result

    @pytest.mark.asyncio
    async def test_basic_code_streams_chunks(self, tool_context):
        """Test that python_repl yields StreamChunk objects during execution."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="print('hello from python')",
                tool_context=tool_context,
            )
        )
        stdout_chunks = [c for c in chunks if c.stream_type == "stdout"]
        assert len(stdout_chunks) >= 1
        assert any("hello from python" in c.data for c in stdout_chunks)

    @pytest.mark.asyncio
    async def test_stderr_streams_as_stderr_chunks(self, tool_context):
        """Test that stderr from Python code yields stderr StreamChunks."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import sys; print('err_msg', file=sys.stderr)",
                tool_context=tool_context,
            )
        )
        stderr_chunks = [c for c in chunks if c.stream_type == "stderr"]
        assert len(stderr_chunks) >= 1
        assert any("err_msg" in c.data for c in stderr_chunks)

    @pytest.mark.asyncio
    async def test_code_with_math(self, tool_context):
        """Test Python math execution."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="print(2 + 2)",
                tool_context=tool_context,
            )
        )
        assert "4" in result

    @pytest.mark.asyncio
    async def test_code_with_error(self, tool_context):
        """Test Python code that raises an error."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="raise ValueError('test error')",
                tool_context=tool_context,
            )
        )
        assert "test error" in result or "ValueError" in result

    @pytest.mark.asyncio
    async def test_code_with_import(self, tool_context):
        """Test Python code with imports."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import json; print(json.dumps({'key': 'value'}))",
                tool_context=tool_context,
            )
        )
        assert "key" in result

    @pytest.mark.asyncio
    async def test_timeout(self, tool_context):
        """Test code execution timeout."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import time; time.sleep(10)",
                timeout=1,
                tool_context=tool_context,
            )
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_config_timeout(self, tool_context, mock_agent):
        """Test timeout from config."""
        mock_agent.state.set("strands_python_repl_tool", {"timeout": 1})
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import time; time.sleep(10)",
                tool_context=tool_context,
            )
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_reset(self, tool_context):
        """Test REPL reset."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="",
                reset=True,
                tool_context=tool_context,
            )
        )
        assert "reset" in result.lower()

    @pytest.mark.asyncio
    async def test_multiline_code(self, tool_context):
        """Test multiline Python code."""
        code = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(10))
"""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code=code,
                tool_context=tool_context,
            )
        )
        assert "55" in result

    @pytest.mark.asyncio
    async def test_no_output(self, tool_context):
        """Test code with no output."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="x = 42",
                tool_context=tool_context,
            )
        )
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_noop_sandbox(self, tool_context, mock_agent):
        """Test python_repl with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="print('test')",
                tool_context=tool_context,
            )
        )
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_stream_chunk_types_are_correct(self, tool_context):
        """Test that all yielded chunks are proper StreamChunk instances."""
        chunks, result = await collect_generator(
            python_repl.__wrapped__(
                code="import sys; print('out'); print('err', file=sys.stderr)",
                tool_context=tool_context,
            )
        )
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)
            assert hasattr(chunk, "data")
            assert hasattr(chunk, "stream_type")
            assert chunk.stream_type in ("stdout", "stderr")
