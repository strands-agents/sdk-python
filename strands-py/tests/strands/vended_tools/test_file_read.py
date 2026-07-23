"""Tests for the sandbox-routed, read-only file tool.

``file_read`` is a thin shim over ``file_editor``'s ``view`` command. The
security surface (absolute paths, ``..`` traversal, size limit, ``view_range``
bounds, missing paths) is enforced by ``file_editor`` and must be reachable
through the shim — these tests confirm that delegation, not re-implementation.
"""

import sys
from types import SimpleNamespace

import pytest

from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.types.tools import ToolContext
from strands.vended_tools.file_read import DEFAULT_FILE_READ_DESCRIPTION, file_read, make_file_read

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path semantics assumed")


def _tool_context(sandbox: NotASandboxLocalEnvironment | None = None) -> ToolContext:
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(
        tool_use={"name": "file_read", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )


def _write(path, content: str) -> str:
    path.write_text(content)
    return str(path)


@pytest.fixture
def reader():
    return make_file_read(sandbox=NotASandboxLocalEnvironment())


@pytest.fixture
def ctx():
    return _tool_context()


class TestReadFile:
    """Happy path: read a whole file or a slice of one."""

    @pytest.mark.asyncio
    async def test_returns_content_with_line_numbers(self, reader, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        result = await reader(path=file_path, tool_context=ctx)
        assert "Here's the result of running `cat -n`" in result
        assert "     1  Line 1" in result
        assert "     3  Line 3" in result

    @pytest.mark.asyncio
    async def test_view_range(self, reader, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        result = await reader(path=file_path, tool_context=ctx, view_range=[2, 4])
        assert "     2  Line 2" in result
        assert "     4  Line 4" in result
        assert "     1  " not in result
        assert "     5  " not in result

    @pytest.mark.asyncio
    async def test_view_range_negative_end(self, reader, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        result = await reader(path=file_path, tool_context=ctx, view_range=[2, -1])
        assert "     2  Line 2" in result
        assert "     3  Line 3" in result
        assert "     1  " not in result

    @pytest.mark.asyncio
    async def test_lists_directory(self, reader, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        _write(d / "file1.txt", "content")
        _write(d / "file2.txt", "content")
        result = await reader(path=str(d), tool_context=ctx)
        assert "file1.txt" in result
        assert "file2.txt" in result


class TestViewRangeShape:
    """The view_range parameter is fixed-length [start, end]; other lengths are rejected."""

    @pytest.mark.asyncio
    async def test_rejects_single_element_range(self, reader, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2")
        agent = SimpleNamespace(sandbox=NotASandboxLocalEnvironment())
        tool_use = {
            "name": "file_read",
            "toolUseId": "malformed",
            "input": {"path": file_path, "view_range": [2]},
        }
        events = [e async for e in reader.stream(tool_use, {"agent": agent})]
        result = events[-1].tool_result
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_rejects_three_element_range(self, reader, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2")
        agent = SimpleNamespace(sandbox=NotASandboxLocalEnvironment())
        tool_use = {
            "name": "file_read",
            "toolUseId": "malformed",
            "input": {"path": file_path, "view_range": [1, 2, 3]},
        }
        events = [e async for e in reader.stream(tool_use, {"agent": agent})]
        result = events[-1].tool_result
        assert result["status"] == "error"


class TestSecurityDelegation:
    """The security surface must be inherited from file_editor verbatim.

    The shim adds no validation logic — file_editor owns the security surface
    (absolute path, ``..`` traversal, size cap, view_range bounds, directory
    checks). We prove delegation with a single traversal probe; the exhaustive
    security suite lives with file_editor.
    """

    @pytest.mark.asyncio
    async def test_surfaces_file_editor_validation(self, reader, ctx):
        with pytest.raises(ValueError, match="path traversal"):
            await reader(path="/tmp/../etc/passwd", tool_context=ctx)


class TestToolMetadata:
    """The tool must expose a narrower, read-only schema — no write parameters."""

    def test_default_name(self):
        assert file_read.tool_name == "file_read"

    def test_custom_name(self):
        assert make_file_read(name="reader").tool_name == "reader"

    def test_default_description(self):
        assert make_file_read().tool_spec["description"] == DEFAULT_FILE_READ_DESCRIPTION

    def test_input_schema_is_read_only(self):
        props = file_read.tool_spec["inputSchema"]["json"]["properties"]
        assert set(props.keys()) == {"path", "view_range"}
        # The write-side surface of file_editor must not leak through.
        for banned in ("command", "file_text", "old_str", "new_str", "insert_line"):
            assert banned not in props

    def test_input_schema_excludes_context(self):
        props = file_read.tool_spec["inputSchema"]["json"]["properties"]
        assert "tool_context" not in props

    @pytest.mark.asyncio
    async def test_smuggled_write_params_are_stripped(self, reader, tmp_path):
        # A model could try to smuggle write-side params through the tool call.
        # Pydantic strips extras from the validated input, so the shim's
        # hard-coded command="view" is the only thing that reaches file_editor.
        file_path = _write(tmp_path / "test.txt", "ok")
        agent = SimpleNamespace(sandbox=NotASandboxLocalEnvironment())
        tool_use = {
            "name": "file_read",
            "toolUseId": "smuggle",
            "input": {"path": file_path, "command": "create", "file_text": "pwned"},
        }
        events = [e async for e in reader.stream(tool_use, {"agent": agent})]
        result = events[-1].tool_result
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Here's the result of running `cat -n`" in text
        assert "ok" in text
        # The smuggled write did not happen.
        assert (tmp_path / "test.txt").read_text() == "ok"
