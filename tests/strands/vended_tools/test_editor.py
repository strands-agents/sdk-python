"""Tests for the editor vended tool."""

import pytest

from strands.sandbox.noop import NoOpSandbox
from strands.vended_tools.editor import editor


class TestEditorTool:
    """Tests for the editor vended tool."""

    @pytest.mark.asyncio
    async def test_view_file(self, tool_context, tmp_path):
        """Test viewing a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            tool_context=tool_context,
        )
        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result
        assert "cat -n" in result

    @pytest.mark.asyncio
    async def test_view_with_range(self, tool_context, tmp_path):
        """Test viewing a file with line range."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            view_range=[2, 4],
            tool_context=tool_context,
        )
        assert "line 2" in result
        assert "line 4" in result
        assert "     1" not in result

    @pytest.mark.asyncio
    async def test_view_with_range_end_minus_one(self, tool_context, tmp_path):
        """Test viewing with -1 as end of range."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            view_range=[2, -1],
            tool_context=tool_context,
        )
        assert "line 2" in result
        assert "line 3" in result

    @pytest.mark.asyncio
    async def test_view_directory(self, tool_context, tmp_path):
        """Test viewing a directory listing."""
        (tmp_path / "file1.py").write_text("pass")
        (tmp_path / "file2.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()

        result = await editor.__wrapped__(
            command="view",
            path=str(tmp_path),
            tool_context=tool_context,
        )
        assert "file1.py" in result
        assert "file2.txt" in result
        assert "subdir/" in result

    @pytest.mark.asyncio
    async def test_view_nonexistent(self, tool_context, tmp_path):
        """Test viewing a nonexistent file."""
        result = await editor.__wrapped__(
            command="view",
            path=str(tmp_path / "nonexistent.txt"),
            tool_context=tool_context,
        )
        assert "does not exist" in result.lower()

    @pytest.mark.asyncio
    async def test_view_invalid_range(self, tool_context, tmp_path):
        """Test viewing with invalid range."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\n")

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            view_range=[0, 2],
            tool_context=tool_context,
        )
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_create_file(self, tool_context, tmp_path):
        """Test creating a new file."""
        new_file = tmp_path / "new_file.py"

        result = await editor.__wrapped__(
            command="create",
            path=str(new_file),
            file_text="print('hello')\n",
            tool_context=tool_context,
        )
        assert "created" in result.lower()
        assert new_file.read_text() == "print('hello')\n"

    @pytest.mark.asyncio
    async def test_create_existing_file(self, tool_context, tmp_path):
        """Test creating a file that already exists."""
        existing = tmp_path / "existing.py"
        existing.write_text("original")

        result = await editor.__wrapped__(
            command="create",
            path=str(existing),
            file_text="new content",
            tool_context=tool_context,
        )
        assert "already exists" in result.lower()
        assert existing.read_text() == "original"

    @pytest.mark.asyncio
    async def test_create_missing_file_text(self, tool_context, tmp_path):
        """Test create without file_text."""
        result = await editor.__wrapped__(
            command="create",
            path=str(tmp_path / "new.py"),
            tool_context=tool_context,
        )
        assert "file_text" in result.lower()

    @pytest.mark.asyncio
    async def test_str_replace_unique(self, tool_context, tmp_path):
        """Test str_replace with a unique match."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'hello'\n")

        result = await editor.__wrapped__(
            command="str_replace",
            path=str(test_file),
            old_str="return 'hello'",
            new_str="return 'world'",
            tool_context=tool_context,
        )
        assert "edited" in result.lower()
        assert "return 'world'" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_str_replace_not_found(self, tool_context, tmp_path):
        """Test str_replace when old_str not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'hello'\n")

        result = await editor.__wrapped__(
            command="str_replace",
            path=str(test_file),
            old_str="nonexistent string",
            new_str="replacement",
            tool_context=tool_context,
        )
        assert "did not appear" in result.lower()

    @pytest.mark.asyncio
    async def test_str_replace_multiple_occurrences(self, tool_context, tmp_path):
        """Test str_replace rejects multiple occurrences."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\ny = 1\nz = 1\n")

        result = await editor.__wrapped__(
            command="str_replace",
            path=str(test_file),
            old_str="= 1",
            new_str="= 2",
            tool_context=tool_context,
        )
        assert "multiple" in result.lower()
        assert test_file.read_text() == "x = 1\ny = 1\nz = 1\n"

    @pytest.mark.asyncio
    async def test_str_replace_deletion(self, tool_context, tmp_path):
        """Test str_replace with empty new_str (deletion)."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: remove this\ndef main():\n    pass\n")

        result = await editor.__wrapped__(
            command="str_replace",
            path=str(test_file),
            old_str="# TODO: remove this\n",
            new_str="",
            tool_context=tool_context,
        )
        assert "edited" in result.lower()
        assert "TODO" not in test_file.read_text()

    @pytest.mark.asyncio
    async def test_insert(self, tool_context, tmp_path):
        """Test inserting text at a line."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 3\n")

        result = await editor.__wrapped__(
            command="insert",
            path=str(test_file),
            insert_line=1,
            new_str="line 2",
            tool_context=tool_context,
        )
        assert "edited" in result.lower()
        content = test_file.read_text()
        assert "line 1\nline 2\nline 3\n" == content

    @pytest.mark.asyncio
    async def test_insert_at_beginning(self, tool_context, tmp_path):
        """Test inserting at the beginning of a file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 2\nline 3\n")

        result = await editor.__wrapped__(
            command="insert",
            path=str(test_file),
            insert_line=0,
            new_str="line 1",
            tool_context=tool_context,
        )
        assert "edited" in result.lower()
        assert test_file.read_text().startswith("line 1\n")

    @pytest.mark.asyncio
    async def test_insert_invalid_line(self, tool_context, tmp_path):
        """Test insert with invalid line number."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\n")

        result = await editor.__wrapped__(
            command="insert",
            path=str(test_file),
            insert_line=999,
            new_str="new line",
            tool_context=tool_context,
        )
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_undo_edit(self, tool_context, tmp_path):
        """Test undo_edit reverting a str_replace."""
        test_file = tmp_path / "test.py"
        test_file.write_text("original content\n")

        await editor.__wrapped__(
            command="str_replace",
            path=str(test_file),
            old_str="original content",
            new_str="modified content",
            tool_context=tool_context,
        )
        assert "modified content" in test_file.read_text()

        result = await editor.__wrapped__(
            command="undo_edit",
            path=str(test_file),
            tool_context=tool_context,
        )
        assert "reverted" in result.lower()
        assert "original content" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_undo_no_history(self, tool_context, tmp_path):
        """Test undo_edit when no history exists."""
        result = await editor.__wrapped__(
            command="undo_edit",
            path=str(tmp_path / "nonexistent.py"),
            tool_context=tool_context,
        )
        assert "no edit history" in result.lower()

    @pytest.mark.asyncio
    async def test_relative_path_allowed_by_default(self, tool_context, tmp_path):
        """Test that relative paths are passed through to sandbox by default."""
        test_file = tmp_path / "relative_test.txt"
        test_file.write_text("relative content\n")

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            tool_context=tool_context,
        )
        assert "relative content" in result

    @pytest.mark.asyncio
    async def test_relative_path_rejected_when_configured(self, tool_context, mock_agent):
        """Test that relative paths are rejected when require_absolute_paths is True."""
        mock_agent.state.set("strands_editor_tool", {"require_absolute_paths": True})

        result = await editor.__wrapped__(
            command="view",
            path="relative/path.py",
            tool_context=tool_context,
        )
        assert "not an absolute path" in result.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_allowed_by_default(self, tool_context, tmp_path):
        """Test that paths with .. are passed through to sandbox by default."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        test_file = tmp_path / "traversal_test.txt"
        test_file.write_text("traversal content\n")

        traversal_path = str(subdir / ".." / "traversal_test.txt")
        result = await editor.__wrapped__(
            command="view",
            path=traversal_path,
            tool_context=tool_context,
        )
        assert "traversal content" in result

    @pytest.mark.asyncio
    async def test_path_traversal_rejected_when_configured(self, tool_context, mock_agent):
        """Test that path traversal is rejected when require_absolute_paths is True."""
        mock_agent.state.set("strands_editor_tool", {"require_absolute_paths": True})

        result = await editor.__wrapped__(
            command="view",
            path="/tmp/../etc/passwd",
            tool_context=tool_context,
        )
        assert "not allowed" in result.lower()

    @pytest.mark.asyncio
    async def test_noop_sandbox(self, tool_context, mock_agent, tmp_path):
        """Test editor with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()

        result = await editor.__wrapped__(
            command="view",
            path=str(tmp_path / "test.py"),
            tool_context=tool_context,
        )
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_max_file_size(self, tool_context, mock_agent, tmp_path):
        """Test max file size configuration."""
        mock_agent.state.set("strands_editor_tool", {"max_file_size": 10})

        test_file = tmp_path / "large.txt"
        test_file.write_text("a" * 100)

        result = await editor.__wrapped__(
            command="view",
            path=str(test_file),
            tool_context=tool_context,
        )
        assert "exceeds" in result.lower()
