"""Tests for the sandbox-routed file editor tool.

Tests for the sandbox-routed file editor tool.
The tool is exercised against a real ``NotASandboxLocalEnvironment`` (host
filesystem), and called directly
(like a normal async function). Errors
surface as raised ``ValueError`` (the raw function raises; the error->status
wrapping only happens through the tool's ``stream`` path). Path semantics assume
POSIX, so these are skipped on Windows.
"""

import sys
from types import SimpleNamespace

import pytest

from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.types.tools import ToolContext
from strands.vended_tools.file_editor import file_editor, make_file_editor
from strands.vended_tools.file_editor.file_editor import DEFAULT_FILE_EDITOR_DESCRIPTION

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path semantics assumed")


def _tool_context(sandbox: NotASandboxLocalEnvironment | None = None) -> ToolContext:
    """Build a ToolContext whose agent exposes the given sandbox (or a fresh one)."""
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(
        tool_use={"name": "file_editor", "toolUseId": "test-id", "input": {}},
        agent=agent,
        invocation_state={},
    )


def _write(path, content: str) -> str:
    """Write content to a path and return it as a string (test helper)."""
    path.write_text(content)
    return str(path)


@pytest.fixture
def editor():
    """A file editor bound to a host sandbox, bound to a host sandbox."""
    return make_file_editor(sandbox=NotASandboxLocalEnvironment())


@pytest.fixture
def ctx():
    return _tool_context()


class TestViewFile:
    """Viewing an entire file or a line range."""

    @pytest.mark.asyncio
    async def test_returns_content_with_line_numbers(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "Here's the result of running `cat -n`" in result
        assert "     1  Line 1" in result
        assert "     2  Line 2" in result
        assert "     3  Line 3" in result

    @pytest.mark.asyncio
    async def test_handles_empty_file(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "empty.txt", "")
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "Here's the result of running `cat -n`" in result

    @pytest.mark.asyncio
    async def test_handles_single_line(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "single.txt", "Only one line")
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "     1  Only one line" in result

    @pytest.mark.asyncio
    async def test_range_returns_specified_lines(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        result = await editor(command="view", path=file_path, tool_context=ctx, view_range=[2, 4])
        assert "     2  Line 2" in result
        assert "     3  Line 3" in result
        assert "     4  Line 4" in result
        assert "     1  " not in result
        assert "     5  " not in result

    @pytest.mark.asyncio
    async def test_range_negative_end_means_to_end(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        result = await editor(command="view", path=file_path, tool_context=ctx, view_range=[3, -1])
        assert "     3  Line 3" in result
        assert "     5  Line 5" in result
        assert "     1  " not in result
        assert "     2  " not in result

    @pytest.mark.asyncio
    async def test_range_single_line(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        result = await editor(command="view", path=file_path, tool_context=ctx, view_range=[2, 2])
        assert "     2  Line 2" in result
        assert "     1  " not in result
        assert "     3  " not in result


class TestViewDirectory:
    """Viewing a directory lists its contents."""

    @pytest.mark.asyncio
    async def test_lists_two_levels_deep(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        (d / "subdir" / "nested").mkdir(parents=True)
        _write(d / "file1.txt", "content")
        _write(d / "file2.txt", "content")
        _write(d / "subdir" / "file3.txt", "content")
        _write(d / "subdir" / "nested" / "file4.txt", "content")
        result = await editor(command="view", path=str(d), tool_context=ctx)
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result
        assert "file3.txt" in result
        assert "file4.txt" in result

    @pytest.mark.asyncio
    async def test_excludes_hidden(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        _write(d / "visible.txt", "content")
        _write(d / ".hidden.txt", "content")
        result = await editor(command="view", path=str(d), tool_context=ctx)
        assert "visible.txt" in result
        assert ".hidden" not in result


class TestViewErrors:
    """Error cases for the view command."""

    @pytest.mark.asyncio
    async def test_nonexistent_raises(self, editor, ctx, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            await editor(command="view", path=str(tmp_path / "nope.txt"), tool_context=ctx)

    @pytest.mark.asyncio
    async def test_relative_path_raises(self, editor, ctx):
        with pytest.raises(ValueError, match="not an absolute path"):
            await editor(command="view", path="relative/path.txt", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_path_traversal_raises(self, editor, ctx):
        with pytest.raises(ValueError, match="path traversal"):
            await editor(command="view", path="/tmp/../etc/passwd", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_range_invalid_start_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        with pytest.raises(ValueError, match="view_range"):
            await editor(command="view", path=file_path, tool_context=ctx, view_range=[0, 2])

    @pytest.mark.asyncio
    async def test_range_end_beyond_length_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        with pytest.raises(ValueError, match="view_range"):
            await editor(command="view", path=file_path, tool_context=ctx, view_range=[1, 10])

    @pytest.mark.asyncio
    async def test_range_end_before_start_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        with pytest.raises(ValueError, match="view_range"):
            await editor(command="view", path=file_path, tool_context=ctx, view_range=[3, 1])

    @pytest.mark.asyncio
    async def test_range_on_directory_raises(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        _write(d / "file.txt", "content")
        with pytest.raises(ValueError, match="not allowed when"):
            await editor(command="view", path=str(d), tool_context=ctx, view_range=[1, 2])


class TestCreate:
    """The create command."""

    @pytest.mark.asyncio
    async def test_new_file(self, editor, ctx, tmp_path):
        file_path = str(tmp_path / "new-file.txt")
        content = "Hello World\nLine 2"
        result = await editor(command="create", path=file_path, tool_context=ctx, file_text=content)
        assert "File created successfully" in result
        assert file_path in result
        assert (tmp_path / "new-file.txt").read_text() == content

    @pytest.mark.asyncio
    async def test_in_nonexistent_directory(self, editor, ctx, tmp_path):
        file_path = str(tmp_path / "newdir" / "subdir" / "new-file.txt")
        result = await editor(command="create", path=file_path, tool_context=ctx, file_text="Content")
        assert "File created successfully" in result
        assert (tmp_path / "newdir" / "subdir" / "new-file.txt").read_text() == "Content"

    @pytest.mark.asyncio
    async def test_empty_file(self, editor, ctx, tmp_path):
        file_path = str(tmp_path / "empty.txt")
        result = await editor(command="create", path=file_path, tool_context=ctx, file_text="")
        assert "File created successfully" in result
        assert (tmp_path / "empty.txt").read_text() == ""

    @pytest.mark.asyncio
    async def test_existing_file_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "existing.txt", "content")
        with pytest.raises(ValueError, match="already exists"):
            await editor(command="create", path=file_path, tool_context=ctx, file_text="new content")

    @pytest.mark.asyncio
    async def test_relative_path_raises(self, editor, ctx):
        with pytest.raises(ValueError, match="not an absolute path"):
            await editor(command="create", path="relative/path.txt", tool_context=ctx, file_text="content")

    @pytest.mark.asyncio
    async def test_on_directory_raises(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        with pytest.raises(ValueError, match="already exists"):
            await editor(command="create", path=str(d), tool_context=ctx, file_text="content")


class TestStrReplace:
    """The str_replace command."""

    @pytest.mark.asyncio
    async def test_unique_occurrence(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2 OLD\nLine 3\nLine 4")
        result = await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="OLD", new_str="NEW")
        assert "has been edited" in result
        assert "NEW" in result
        assert (tmp_path / "test.txt").read_text() == "Line 1\nLine 2 NEW\nLine 3\nLine 4"

    @pytest.mark.asyncio
    async def test_snippet_window(self, editor, ctx, tmp_path):
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5 OLD\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10"
        file_path = _write(tmp_path / "test.txt", content)
        result = await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="OLD", new_str="NEW")
        assert "Line 1" in result
        assert "Line 9" in result
        assert "Line 10" not in result

    @pytest.mark.asyncio
    async def test_deletion(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2 DELETE_ME\nLine 3")
        result = await editor(command="str_replace", path=file_path, tool_context=ctx, old_str=" DELETE_ME", new_str="")
        assert "has been edited" in result
        assert (tmp_path / "test.txt").read_text() == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_multiline(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nOLD LINE 1\nOLD LINE 2\nLine 4")
        await editor(
            command="str_replace",
            path=file_path,
            tool_context=ctx,
            old_str="OLD LINE 1\nOLD LINE 2",
            new_str="NEW LINE",
        )
        assert (tmp_path / "test.txt").read_text() == "Line 1\nNEW LINE\nLine 4"

    @pytest.mark.asyncio
    async def test_preserves_dollar_patterns_literally(self, editor, ctx, tmp_path):
        # Python's str.replace is literal, so $&/$1/$$ must survive verbatim
        file_path = _write(tmp_path / "test.txt", "const value = getPrice()")
        await editor(
            command="str_replace", path=file_path, tool_context=ctx, old_str="getPrice()", new_str="$& is not $1 or $$"
        )
        assert (tmp_path / "test.txt").read_text() == "const value = $& is not $1 or $$"

    @pytest.mark.asyncio
    async def test_not_found_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        with pytest.raises(ValueError, match="did not appear"):
            await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="NOTFOUND", new_str="NEW")

    @pytest.mark.asyncio
    async def test_multiple_occurrences_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "DUP Line 1\nLine 2\nDUP Line 3")
        with pytest.raises(ValueError, match="Multiple occurrences"):
            await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="DUP", new_str="NEW")

    @pytest.mark.asyncio
    async def test_nonexistent_raises(self, editor, ctx, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            await editor(
                command="str_replace", path=str(tmp_path / "nope.txt"), tool_context=ctx, old_str="OLD", new_str="NEW"
            )

    @pytest.mark.asyncio
    async def test_on_directory_raises(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        with pytest.raises(ValueError, match="directory"):
            await editor(command="str_replace", path=str(d), tool_context=ctx, old_str="OLD", new_str="NEW")


class TestInsert:
    """The insert command."""

    @pytest.mark.asyncio
    async def test_at_beginning(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=0, new_str="NEW LINE")
        assert (tmp_path / "test.txt").read_text() == "NEW LINE\nLine 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_in_middle(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=2, new_str="NEW LINE")
        assert (tmp_path / "test.txt").read_text() == "Line 1\nLine 2\nNEW LINE\nLine 3"

    @pytest.mark.asyncio
    async def test_at_end(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2\nLine 3")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=3, new_str="NEW LINE")
        assert (tmp_path / "test.txt").read_text() == "Line 1\nLine 2\nLine 3\nNEW LINE"

    @pytest.mark.asyncio
    async def test_snippet_window(self, editor, ctx, tmp_path):
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9"
        file_path = _write(tmp_path / "test.txt", content)
        result = await editor(command="insert", path=file_path, tool_context=ctx, insert_line=5, new_str="INSERTED")
        assert "Line 2" in result
        assert "Line 9" in result
        assert "INSERTED" in result

    @pytest.mark.asyncio
    async def test_multiline(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=1, new_str="NEW 1\nNEW 2\nNEW 3")
        assert (tmp_path / "test.txt").read_text() == "Line 1\nNEW 1\nNEW 2\nNEW 3\nLine 2"

    @pytest.mark.asyncio
    async def test_in_empty_file(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "empty.txt", "")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=0, new_str="First line")
        assert (tmp_path / "empty.txt").read_text() == "First line"

    @pytest.mark.asyncio
    async def test_negative_line_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2")
        with pytest.raises(ValueError, match="insert_line"):
            await editor(command="insert", path=file_path, tool_context=ctx, insert_line=-1, new_str="NEW")

    @pytest.mark.asyncio
    async def test_beyond_length_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "Line 1\nLine 2")
        with pytest.raises(ValueError, match="insert_line"):
            await editor(command="insert", path=file_path, tool_context=ctx, insert_line=10, new_str="NEW")

    @pytest.mark.asyncio
    async def test_nonexistent_raises(self, editor, ctx, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            await editor(
                command="insert", path=str(tmp_path / "nope.txt"), tool_context=ctx, insert_line=0, new_str="NEW"
            )

    @pytest.mark.asyncio
    async def test_on_directory_raises(self, editor, ctx, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        with pytest.raises(ValueError, match="directory"):
            await editor(command="insert", path=str(d), tool_context=ctx, insert_line=0, new_str="NEW")


class TestFileSizeLimit:
    """The configurable content size guard."""

    @pytest.mark.asyncio
    async def test_view_exceeds_size_limit_raises(self, ctx, tmp_path):
        # Use a low custom cap so the test doesn't need to allocate the default cap.
        small_editor = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=1024)
        file_path = _write(tmp_path / "large.txt", "x" * 2048)
        with pytest.raises(ValueError, match="exceeds"):
            await small_editor(command="view", path=file_path, tool_context=ctx)

    @pytest.mark.asyncio
    async def test_default_cap_is_one_megabyte(self, editor, ctx, tmp_path):
        # Default cap is 1 MB. A file just under it is accepted; a 2 MB file is rejected.
        under = _write(tmp_path / "under.txt", "x" * (1 * 1024 * 1024 - 1))
        result = await editor(command="view", path=under, tool_context=ctx)
        assert "cat -n" in result
        over = _write(tmp_path / "over.txt", "x" * (2 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeds"):
            await editor(command="view", path=over, tool_context=ctx)


class TestEdgeCases:
    """Content edge cases: special characters, unicode, tabs, trailing slashes."""

    @pytest.mark.asyncio
    async def test_special_characters(self, editor, ctx, tmp_path):
        content = 'Special chars: @#$%^&*()_+-={}[]|:;"<>,.?/~`'
        file_path = _write(tmp_path / "special.txt", content)
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "Special chars:" in result

    @pytest.mark.asyncio
    async def test_unicode(self, editor, ctx, tmp_path):
        content = "你好世界\n🚀 Emoji test\nΣ Greek letters"
        file_path = _write(tmp_path / "unicode.txt", content)
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "你好世界" in result
        assert "🚀" in result

    @pytest.mark.asyncio
    async def test_expands_tabs(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "tabs.txt", "Line 1\tTab\tSeparated")
        result = await editor(command="view", path=file_path, tool_context=ctx)
        assert "\t" not in result

    @pytest.mark.asyncio
    async def test_handles_trailing_slash_on_file_path(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "trailing.txt", "content here")
        result = await editor(command="view", path=f"{file_path}/", tool_context=ctx)
        assert "content here" in result


class TestSandboxErrorPropagation:
    """A non-'not found' listing error must propagate, not be disguised as non-existence."""

    @pytest.mark.asyncio
    async def test_propagates_non_not_found_list_errors(self):
        sandbox = NotASandboxLocalEnvironment()

        async def boom(path, **kwargs):
            raise OSError("EACCES: permission denied")

        sandbox.list_files = boom  # type: ignore[method-assign]
        editor = make_file_editor(sandbox=sandbox)
        with pytest.raises(OSError, match="permission denied"):
            await editor(command="view", path="/tmp/x.txt", tool_context=_tool_context(sandbox))


class TestToolMetadata:
    """Tests for the file editor tool's name, description, and input schema."""

    def test_default_name(self):
        assert file_editor.tool_name == "file_editor"

    def test_custom_name(self):
        assert make_file_editor(name="sandbox_file_editor").tool_name == "sandbox_file_editor"

    def test_default_description(self):
        assert make_file_editor().tool_spec["description"] == DEFAULT_FILE_EDITOR_DESCRIPTION

    def test_input_schema_excludes_context(self):
        props = file_editor.tool_spec["inputSchema"]["json"]["properties"]
        assert "command" in props
        assert "path" in props
        assert "tool_context" not in props

    def test_input_schema_advertises_new_commands(self):
        # The literal-enum values must include the newly reconciled commands so
        # models see them in the tool schema.
        command_enum = file_editor.tool_spec["inputSchema"]["json"]["properties"]["command"]["enum"]
        assert set(command_enum) == {
            "view",
            "create",
            "str_replace",
            "insert",
            "pattern_replace",
            "find_line",
            "undo_edit",
        }


class TestConfinementRoot:
    """When a ``root`` is configured, all paths must resolve inside it."""

    @pytest.mark.asyncio
    async def test_rejects_absolute_path_outside_root(self, ctx, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        _write(outside, "secret")
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        with pytest.raises(ValueError, match="outside the configured root"):
            await confined(command="view", path=str(outside), tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_traversal_even_with_root(self, ctx, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        with pytest.raises(ValueError, match="path traversal"):
            await confined(command="view", path=f"{root}/../outside.txt", tool_context=ctx)

    @pytest.mark.asyncio
    async def test_rejects_sibling_that_shares_prefix(self, ctx, tmp_path):
        root = tmp_path / "ws"
        root.mkdir()
        sibling = tmp_path / "ws-neighbor"
        sibling.mkdir()
        _write(sibling / "file.txt", "content")
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        with pytest.raises(ValueError, match="outside the configured root"):
            await confined(command="view", path=str(sibling / "file.txt"), tool_context=ctx)

    @pytest.mark.asyncio
    async def test_allows_path_inside_root(self, ctx, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        target = _write(root / "ok.txt", "hello")
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        result = await confined(command="view", path=target, tool_context=ctx)
        assert "hello" in result

    def test_rejects_relative_root_at_construction(self):
        with pytest.raises(ValueError, match="absolute path"):
            make_file_editor(root="relative/root")

    @pytest.mark.asyncio
    async def test_symlink_pointing_outside_root_is_rejected(self, ctx, tmp_path):
        # A symlink inside root that resolves to a file outside root must not
        # slip past the string-level confinement check.
        root = tmp_path / "workspace"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        _write(secret, "top secret")
        link = root / "escape.txt"
        link.symlink_to(secret)
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        with pytest.raises(ValueError, match="symlink|outside"):
            await confined(command="view", path=str(link), tool_context=ctx)

    @pytest.mark.asyncio
    async def test_symlink_pointing_inside_root_is_allowed(self, ctx, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        target = _write(root / "real.txt", "inside content")
        link = root / "alias.txt"
        link.symlink_to(target)
        confined = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=str(root))
        result = await confined(command="view", path=str(link), tool_context=ctx)
        assert "inside content" in result


class TestWriteSizeCaps:
    """The write side must also reject payloads above the configured cap."""

    @pytest.mark.asyncio
    async def test_create_rejects_oversize_file_text(self, ctx, tmp_path):
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=1024)
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await e(
                command="create",
                path=str(tmp_path / "big.txt"),
                tool_context=ctx,
                file_text="x" * 2048,
            )

    @pytest.mark.asyncio
    async def test_str_replace_rejects_oversize_new_str(self, ctx, tmp_path):
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=1024)
        file_path = _write(tmp_path / "s.txt", "small")
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            await e(
                command="str_replace",
                path=file_path,
                tool_context=ctx,
                old_str="small",
                new_str="y" * 2048,
            )


class TestNonLocalRoot:
    """A ``root`` that does not exist on the local host (e.g. Docker sandbox) must not
    fail every op via ``realpath``. The realpath layer must be skipped; the string-
    level confinement still applies."""

    @pytest.mark.asyncio
    async def test_confines_container_side_root(self, ctx):
        # A container-side path with no local counterpart. The tool should
        # accept an in-root path (and fail later at the sandbox with a
        # not-found), and reject an out-of-root path at the string level.
        container_root = "/workspace-in-container-does-not-exist-locally"
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), root=container_root)
        with pytest.raises(ValueError, match="does not exist"):
            await e(command="view", path=f"{container_root}/foo.txt", tool_context=ctx)
        with pytest.raises(ValueError, match="outside the configured root"):
            await e(command="view", path="/etc/passwd", tool_context=ctx)


class TestPatternReplaceReDoSGuards:
    """``pattern_replace`` must bound pattern length, match count, and wall-clock time."""

    @pytest.mark.asyncio
    async def test_rejects_overlong_pattern(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "content")
        with pytest.raises(ValueError, match="exceeds maximum"):
            await editor(
                command="pattern_replace",
                path=file_path,
                tool_context=ctx,
                pattern="a" * 1001,
                new_str="x",
            )

    @pytest.mark.asyncio
    async def test_rejects_catastrophic_regex_via_timeout(self, ctx, tmp_path):
        # (a+)+b against an all-a input is the canonical catastrophic
        # backtracking example. Use a small input plus a short timeout so the
        # test exercises the timeout path in ~0.1s instead of stalling pytest.
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), pattern_replace_timeout=0.05)
        file_path = _write(tmp_path / "evil.txt", "a" * 24)
        with pytest.raises(ValueError, match="timed out|catastrophic"):
            await e(
                command="pattern_replace",
                path=file_path,
                tool_context=ctx,
                pattern=r"(a+)+b",
                new_str="x",
            )


class TestUndoLRUEviction:
    """The undo history is a bounded LRU: it must evict on overflow."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_entry_past_entry_cap(self, ctx, tmp_path):
        # Two-entry cap; three edits must evict the oldest.
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_undo_entries=2)
        paths = [_write(tmp_path / f"f{i}.txt", f"orig{i}") for i in range(3)]
        for i, p in enumerate(paths):
            await e(command="str_replace", path=p, tool_context=ctx, old_str=f"orig{i}", new_str=f"new{i}")
        # Oldest snapshot (f0) has been evicted.
        with pytest.raises(ValueError, match="No undo history"):
            await e(command="undo_edit", path=paths[0], tool_context=ctx)
        # Two most recent still restore.
        await e(command="undo_edit", path=paths[1], tool_context=ctx)
        assert (tmp_path / "f1.txt").read_text() == "orig1"

    @pytest.mark.asyncio
    async def test_evicts_past_byte_cap(self, ctx, tmp_path):
        # Byte cap sized so one snapshot fits but a second forces eviction of
        # the oldest. Mirrors the entry-cap test's "keep newest, drop oldest".
        snapshot = "x" * 100
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_undo_bytes=len(snapshot) + 20)
        p1 = _write(tmp_path / "big1.txt", snapshot)
        p2 = _write(tmp_path / "big2.txt", "y" * 100)
        await e(command="str_replace", path=p1, tool_context=ctx, old_str="x", new_str="X", replace_all=True)
        await e(command="str_replace", path=p2, tool_context=ctx, old_str="y", new_str="Y", replace_all=True)
        with pytest.raises(ValueError, match="No undo history"):
            await e(command="undo_edit", path=p1, tool_context=ctx)
        await e(command="undo_edit", path=p2, tool_context=ctx)
        assert (tmp_path / "big2.txt").read_text() == "y" * 100


class TestBinaryRejection:
    """Binary files must be rejected on view/edit rather than silently mangled."""

    @pytest.mark.asyncio
    async def test_view_rejects_binary_file(self, editor, ctx, tmp_path):
        file_path = tmp_path / "binary.bin"
        file_path.write_bytes(b"\x00\x01\x02BINARY\x00DATA")
        with pytest.raises(ValueError, match="binary"):
            await editor(command="view", path=str(file_path), tool_context=ctx)

    @pytest.mark.asyncio
    async def test_str_replace_rejects_binary_file(self, editor, ctx, tmp_path):
        file_path = tmp_path / "binary.bin"
        file_path.write_bytes(b"HEAD\x00TAIL")
        with pytest.raises(ValueError, match="binary"):
            await editor(command="str_replace", path=str(file_path), tool_context=ctx, old_str="HEAD", new_str="X")

    @pytest.mark.asyncio
    async def test_utf16_le_bom_rejected_as_unsupported_encoding(self, editor, ctx, tmp_path):
        # UTF-16 text is a valid, decodable format but not UTF-8. Report it as
        # an unsupported encoding rather than misclassifying as binary.
        file_path = tmp_path / "utf16.txt"
        file_path.write_bytes(b"\xff\xfe" + "hello world".encode("utf-16-le"))
        with pytest.raises(ValueError, match="UTF-16"):
            await editor(command="view", path=str(file_path), tool_context=ctx)


class TestStrReplaceReplaceAll:
    """``str_replace`` must refuse ambiguous matches without ``replace_all``."""

    @pytest.mark.asyncio
    async def test_ambiguous_without_opt_in_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "DUP\nfoo\nDUP\nbar\nDUP\n")
        with pytest.raises(ValueError, match="replace_all"):
            await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="DUP", new_str="X")

    @pytest.mark.asyncio
    async def test_replace_all_opt_in_replaces_every_occurrence(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "DUP\nfoo\nDUP\nbar\nDUP\n")
        result = await editor(
            command="str_replace",
            path=file_path,
            tool_context=ctx,
            old_str="DUP",
            new_str="X",
            replace_all=True,
        )
        assert "3 occurrences replaced" in result
        assert (tmp_path / "test.txt").read_text() == "X\nfoo\nX\nbar\nX\n"


class TestPatternReplace:
    """The ``pattern_replace`` regex command."""

    @pytest.mark.asyncio
    async def test_unique_match(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "hello world\ngoodbye")
        await editor(
            command="pattern_replace",
            path=file_path,
            tool_context=ctx,
            pattern=r"hello (\w+)",
            new_str=r"HI \1",
        )
        assert (tmp_path / "test.txt").read_text() == "HI world\ngoodbye"

    @pytest.mark.asyncio
    async def test_ambiguous_without_replace_all_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "foo\nfoo\nfoo\n")
        with pytest.raises(ValueError, match="replace_all"):
            await editor(command="pattern_replace", path=file_path, tool_context=ctx, pattern="foo", new_str="bar")

    @pytest.mark.asyncio
    async def test_replace_all_opt_in(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "foo\nfoo\nfoo\n")
        await editor(
            command="pattern_replace",
            path=file_path,
            tool_context=ctx,
            pattern="foo",
            new_str="bar",
            replace_all=True,
        )
        assert (tmp_path / "test.txt").read_text() == "bar\nbar\nbar\n"

    @pytest.mark.asyncio
    async def test_invalid_pattern_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "content")
        with pytest.raises(ValueError, match="Invalid regex"):
            await editor(command="pattern_replace", path=file_path, tool_context=ctx, pattern="[unclosed", new_str="x")

    @pytest.mark.asyncio
    async def test_no_match_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "content")
        with pytest.raises(ValueError, match="did not match"):
            await editor(command="pattern_replace", path=file_path, tool_context=ctx, pattern="MISSING", new_str="x")


class TestFindLine:
    """The ``find_line`` command."""

    @pytest.mark.asyncio
    async def test_finds_first_occurrence(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "alpha\nbeta\ngamma\nbeta again\n")
        result = await editor(command="find_line", path=file_path, tool_context=ctx, search_text="beta")
        assert "line 2" in result
        assert "gamma" in result  # context snippet

    @pytest.mark.asyncio
    async def test_not_found_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "alpha\n")
        with pytest.raises(ValueError, match="Could not find"):
            await editor(command="find_line", path=file_path, tool_context=ctx, search_text="MISSING")

    @pytest.mark.asyncio
    async def test_fuzzy_matches_across_whitespace(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "def   my_function ( ):\n    pass\n")
        result = await editor(
            command="find_line", path=file_path, tool_context=ctx, search_text="def my_function", fuzzy=True
        )
        assert "line 1" in result


class TestUndo:
    """In-memory, per-tool-instance ``undo_edit``."""

    @pytest.mark.asyncio
    async def test_undo_str_replace(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "hello\n")
        await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="hello", new_str="goodbye")
        assert (tmp_path / "test.txt").read_text() == "goodbye\n"
        await editor(command="undo_edit", path=file_path, tool_context=ctx)
        assert (tmp_path / "test.txt").read_text() == "hello\n"

    @pytest.mark.asyncio
    async def test_undo_insert(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "one\ntwo\n")
        await editor(command="insert", path=file_path, tool_context=ctx, insert_line=1, new_str="between")
        await editor(command="undo_edit", path=file_path, tool_context=ctx)
        assert (tmp_path / "test.txt").read_text() == "one\ntwo\n"

    @pytest.mark.asyncio
    async def test_undo_pattern_replace(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "a\nb\nc\n")
        await editor(command="pattern_replace", path=file_path, tool_context=ctx, pattern="a", new_str="A")
        await editor(command="undo_edit", path=file_path, tool_context=ctx)
        assert (tmp_path / "test.txt").read_text() == "a\nb\nc\n"

    @pytest.mark.asyncio
    async def test_undo_without_history_raises(self, editor, ctx, tmp_path):
        file_path = _write(tmp_path / "test.txt", "content\n")
        with pytest.raises(ValueError, match="No undo history"):
            await editor(command="undo_edit", path=file_path, tool_context=ctx)

    @pytest.mark.asyncio
    async def test_undo_is_scoped_to_tool_instance(self, ctx, tmp_path):
        # Two independent tool instances share no history.
        file_path = _write(tmp_path / "test.txt", "original\n")
        editor_a = make_file_editor(sandbox=NotASandboxLocalEnvironment())
        editor_b = make_file_editor(sandbox=NotASandboxLocalEnvironment())
        await editor_a(command="str_replace", path=file_path, tool_context=ctx, old_str="original", new_str="changed")
        with pytest.raises(ValueError, match="No undo history"):
            await editor_b(command="undo_edit", path=file_path, tool_context=ctx)


class TestEmptyInputRejection:
    """Empty ``old_str`` / ``pattern`` inputs produce confusing behavior and are rejected."""

    @pytest.mark.asyncio
    async def test_str_replace_empty_old_str_raises(self, editor, ctx, tmp_path):
        # `"".count("")` returns `len + 1`; without an explicit guard the caller
        # would see a confusing "multiple occurrences" error or, with
        # replace_all, `new_str` inserted between every character.
        file_path = _write(tmp_path / "test.txt", "hello\n")
        with pytest.raises(ValueError, match="empty"):
            await editor(command="str_replace", path=file_path, tool_context=ctx, old_str="", new_str="X")

    @pytest.mark.asyncio
    async def test_pattern_replace_empty_pattern_raises(self, editor, ctx, tmp_path):
        # An empty pattern compiles to a zero-width regex that matches at every
        # position — surprising, and almost certainly a caller error.
        file_path = _write(tmp_path / "test.txt", "hello\n")
        with pytest.raises(ValueError, match="empty"):
            await editor(command="pattern_replace", path=file_path, tool_context=ctx, pattern="", new_str="x")


class TestPatternReplaceBadBackreference:
    """Bad ``new_str`` backreferences must surface as a clean ``ValueError``."""

    @pytest.mark.asyncio
    async def test_missing_group_backreference_raises_valueerror(self, editor, ctx, tmp_path):
        # `re.sub` raises `re.error` at substitution time when new_str references
        # a group that doesn't exist. The worker-thread wrapper must convert
        # this into a caller-friendly ValueError rather than leaking as an
        # internal exception.
        file_path = _write(tmp_path / "test.txt", "hello world\n")
        with pytest.raises(ValueError, match="Invalid replacement"):
            await editor(
                command="pattern_replace",
                path=file_path,
                tool_context=ctx,
                pattern=r"hello",
                new_str=r"\9",
            )


class TestOversizePostEdit:
    """An edit whose *result* is larger than ``max_file_size`` is rejected before write."""

    @pytest.mark.asyncio
    async def test_str_replace_expansion_past_cap_rejected(self, ctx, tmp_path):
        # Cap is 32 bytes. Original is 20 bytes. `replace_all` on "x" -> "XXX"
        # produces 60 bytes, past the cap.
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=32)
        file_path = _write(tmp_path / "grow.txt", "x" * 20)
        with pytest.raises(ValueError, match="exceeding the maximum"):
            await e(
                command="str_replace",
                path=file_path,
                tool_context=ctx,
                old_str="x",
                new_str="XXX",
                replace_all=True,
            )

    @pytest.mark.asyncio
    async def test_insert_expansion_past_cap_rejected(self, ctx, tmp_path):
        # Original is 30 bytes. `new_str` is 10 bytes. The result would be 41
        # bytes ("\n" adds one on insert into non-empty content), past the cap.
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=32)
        file_path = _write(tmp_path / "grow.txt", "x" * 30)
        with pytest.raises(ValueError, match="exceeding the maximum"):
            await e(
                command="insert",
                path=file_path,
                tool_context=ctx,
                insert_line=0,
                new_str="y" * 10,
            )

    @pytest.mark.asyncio
    async def test_pattern_replace_expansion_past_cap_rejected(self, ctx, tmp_path):
        e = make_file_editor(sandbox=NotASandboxLocalEnvironment(), max_file_size=32)
        file_path = _write(tmp_path / "grow.txt", "x" * 20)
        with pytest.raises(ValueError, match="exceeding the maximum"):
            await e(
                command="pattern_replace",
                path=file_path,
                tool_context=ctx,
                pattern="x",
                new_str="XXX",
                replace_all=True,
            )
