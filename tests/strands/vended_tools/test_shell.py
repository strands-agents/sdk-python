"""Tests for the shell vended tool."""

import pytest

from strands.sandbox.base import StreamChunk
from strands.sandbox.noop import NoOpSandbox
from strands.vended_tools.shell import shell

from .conftest import collect_generator


class TestShellTool:
    """Tests for the shell vended tool."""

    @pytest.mark.asyncio
    async def test_basic_command(self, tool_context, tmp_path):
        """Test basic shell command execution returns result."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo hello", tool_context=tool_context)
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_basic_command_streams_chunks(self, tool_context, tmp_path):
        """Test that shell yields StreamChunk objects during execution."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo hello", tool_context=tool_context)
        )
        # Should have at least one stdout chunk
        stdout_chunks = [c for c in chunks if c.stream_type == "stdout"]
        assert len(stdout_chunks) >= 1
        assert any("hello" in c.data for c in stdout_chunks)
        # Final result should also contain the output
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_stderr_streams_as_stderr_chunks(self, tool_context, tmp_path):
        """Test that stderr output yields StreamChunk with stream_type='stderr'."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo error >&2", tool_context=tool_context)
        )
        stderr_chunks = [c for c in chunks if c.stream_type == "stderr"]
        assert len(stderr_chunks) >= 1
        assert any("error" in c.data for c in stderr_chunks)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_mixed_stdout_stderr_streaming(self, tool_context, tmp_path):
        """Test command with both stdout and stderr streams both chunk types."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command="echo out && echo err >&2",
                tool_context=tool_context,
            )
        )
        stdout_chunks = [c for c in chunks if c.stream_type == "stdout"]
        stderr_chunks = [c for c in chunks if c.stream_type == "stderr"]
        assert len(stdout_chunks) >= 1
        assert len(stderr_chunks) >= 1

    @pytest.mark.asyncio
    async def test_command_with_exit_code(self, tool_context, tmp_path):
        """Test command that returns non-zero exit code."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="exit 42", tool_context=tool_context)
        )
        assert "42" in result

    @pytest.mark.asyncio
    async def test_timeout(self, tool_context):
        """Test command timeout."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="sleep 10", timeout=1, tool_context=tool_context)
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_config_timeout(self, tool_context, mock_agent):
        """Test timeout from config."""
        mock_agent.state.set("strands_shell_tool", {"timeout": 1})
        chunks, result = await collect_generator(
            shell.__wrapped__(command="sleep 10", tool_context=tool_context)
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_restart(self, tool_context):
        """Test shell restart."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="", restart=True, tool_context=tool_context)
        )
        assert "reset" in result.lower()

    @pytest.mark.asyncio
    async def test_no_output_command(self, tool_context):
        """Test command with no output."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="true", tool_context=tool_context)
        )
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_noop_sandbox(self, tool_context, mock_agent):
        """Test shell with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo test", tool_context=tool_context)
        )
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_cwd_tracking(self, tool_context, tmp_path):
        """Test that working directory is tracked across calls."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        chunks, result = await collect_generator(
            shell.__wrapped__(command=f"cd {subdir}", tool_context=tool_context)
        )

        # Verify cwd state was tracked
        shell_state = tool_context.agent.state.get("_strands_shell_state")
        assert shell_state is not None
        assert shell_state["cwd"] == str(subdir)

        # Verify no internal markers leak into the result or streamed chunks
        assert "__STRANDS_CWD__" not in result
        for chunk in chunks:
            assert "__STRANDS_CWD__" not in chunk.data

    @pytest.mark.asyncio
    async def test_cwd_no_marker_leak_in_streaming(self, tool_context, tmp_path):
        """Test that no internal markers appear in streamed output."""
        chunks, result = await collect_generator(
            shell.__wrapped__(command="echo hello && cd /tmp", tool_context=tool_context)
        )
        # No internal markers should appear in any output
        all_chunk_data = "".join(c.data for c in chunks)
        assert "__STRANDS_CWD__" not in all_chunk_data
        assert "__STRANDS_CWD__" not in result

    @pytest.mark.asyncio
    async def test_multiline_output(self, tool_context):
        """Test command with multiline output."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command="echo 'line1\nline2\nline3'",
                tool_context=tool_context,
            )
        )
        assert "line1" in result
        assert "line2" in result

    @pytest.mark.asyncio
    async def test_pipe_command(self, tool_context):
        """Test piped commands."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command="echo 'hello world' | wc -w",
                tool_context=tool_context,
            )
        )
        assert "2" in result

    @pytest.mark.asyncio
    async def test_stream_chunk_types_are_correct(self, tool_context):
        """Test that all yielded chunks are proper StreamChunk instances."""
        chunks, result = await collect_generator(
            shell.__wrapped__(
                command="echo stdout_data && echo stderr_data >&2",
                tool_context=tool_context,
            )
        )
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)
            assert hasattr(chunk, "data")
            assert hasattr(chunk, "stream_type")
            assert chunk.stream_type in ("stdout", "stderr")
