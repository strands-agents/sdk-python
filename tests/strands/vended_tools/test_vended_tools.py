"""Tests for vended tools — shell, editor, python_repl.

These tests use a real HostSandbox to validate end-to-end behavior.
They also test configuration via agent.state and interrupt support.
"""

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from strands.agent.state import AgentState
from strands.sandbox.host import HostSandbox
from strands.sandbox.noop import NoOpSandbox
from strands.types.tools import ToolContext, ToolUse

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sandbox(tmp_path):
    """Create a HostSandbox for testing."""
    return HostSandbox(working_dir=str(tmp_path))


@pytest.fixture
def agent_state():
    """Create a fresh AgentState."""
    return AgentState()


@pytest.fixture
def mock_agent(sandbox, agent_state):
    """Create a mock agent with sandbox and state."""
    agent = MagicMock()
    agent.sandbox = sandbox
    agent.state = agent_state
    agent._interrupt_state = MagicMock()
    agent._interrupt_state.interrupts = {}
    return agent


@pytest.fixture
def tool_use():
    """Create a mock tool use."""
    return ToolUse(
        toolUseId=str(uuid.uuid4()),
        name="test_tool",
        input={},
    )


@pytest.fixture
def tool_context(mock_agent, tool_use):
    """Create a ToolContext for testing."""
    ctx = ToolContext(
        tool_use=tool_use,
        agent=mock_agent,
        invocation_state={},
    )
    return ctx


def run(coro):
    """Helper to run async coroutines in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================
# Shell Tool Tests
# ============================================================


class TestShellTool:
    """Tests for the shell vended tool."""

    def test_basic_command(self, tool_context, tmp_path):
        """Test basic shell command execution."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="echo hello", tool_context=tool_context))
        assert "hello" in result

    def test_command_with_exit_code(self, tool_context, tmp_path):
        """Test command that returns non-zero exit code."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="exit 42", tool_context=tool_context))
        assert "42" in result

    def test_command_with_stderr(self, tool_context, tmp_path):
        """Test command that writes to stderr."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="echo error >&2", tool_context=tool_context))
        assert "error" in result

    def test_timeout(self, tool_context):
        """Test command timeout."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="sleep 10", timeout=1, tool_context=tool_context))
        assert "timed out" in result.lower() or "error" in result.lower()

    def test_config_timeout(self, tool_context, mock_agent):
        """Test timeout from config."""
        from strands.vended_tools.shell.shell import shell

        mock_agent.state.set("strands_shell_tool", {"timeout": 1})
        result = run(shell.__wrapped__(command="sleep 10", tool_context=tool_context))
        assert "timed out" in result.lower() or "error" in result.lower()

    def test_restart(self, tool_context):
        """Test shell restart."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="", restart=True, tool_context=tool_context))
        assert "reset" in result.lower()

    def test_no_output_command(self, tool_context):
        """Test command with no output."""
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="true", tool_context=tool_context))
        assert result == "(no output)"

    def test_noop_sandbox(self, tool_context, mock_agent):
        """Test shell with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()
        from strands.vended_tools.shell.shell import shell

        result = run(shell.__wrapped__(command="echo test", tool_context=tool_context))
        assert "error" in result.lower()

    def test_cwd_tracking(self, tool_context, tmp_path):
        """Test that working directory is tracked across calls."""
        from strands.vended_tools.shell.shell import shell

        # Create a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # cd into it
        run(shell.__wrapped__(command=f"cd {subdir}", tool_context=tool_context))

        # Check tracked state
        shell_state = tool_context.agent.state.get("_strands_shell_state")
        assert shell_state is not None

    def test_multiline_output(self, tool_context):
        """Test command with multiline output."""
        from strands.vended_tools.shell.shell import shell

        result = run(
            shell.__wrapped__(
                command="echo 'line1\nline2\nline3'",
                tool_context=tool_context,
            )
        )
        assert "line1" in result
        assert "line2" in result

    def test_pipe_command(self, tool_context):
        """Test piped commands."""
        from strands.vended_tools.shell.shell import shell

        result = run(
            shell.__wrapped__(
                command="echo 'hello world' | wc -w",
                tool_context=tool_context,
            )
        )
        assert "2" in result

    def test_interrupt_confirmation_approved(self, tool_context, mock_agent):
        """Test interrupt confirmation when approved."""
        from strands.vended_tools.shell.shell import shell

        mock_agent.state.set("strands_shell_tool", {"require_confirmation": True})

        # Mock interrupt to return "approve"
        mock_agent._interrupt_state.interrupts = {}

        # We need to simulate the interrupt mechanism.
        # When require_confirmation is True, the tool calls tool_context.interrupt()
        # which raises InterruptException the first time, then returns the response on resume.
        # For testing, we'll patch the interrupt method.
        with patch.object(type(tool_context), "interrupt", return_value="approve"):
            result = run(shell.__wrapped__(command="echo approved", tool_context=tool_context))
            assert "approved" in result

    def test_interrupt_confirmation_denied(self, tool_context, mock_agent):
        """Test interrupt confirmation when denied."""
        from strands.vended_tools.shell.shell import shell

        mock_agent.state.set("strands_shell_tool", {"require_confirmation": True})

        with patch.object(type(tool_context), "interrupt", return_value="deny"):
            result = run(shell.__wrapped__(command="echo test", tool_context=tool_context))
            assert "not approved" in result.lower()


# ============================================================
# Editor Tool Tests
# ============================================================


class TestEditorTool:
    """Tests for the editor vended tool."""

    def test_view_file(self, tool_context, tmp_path, sandbox):
        """Test viewing a file."""
        from strands.vended_tools.editor.editor import editor

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(test_file),
                tool_context=tool_context,
            )
        )
        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result
        assert "cat -n" in result

    def test_view_with_range(self, tool_context, tmp_path):
        """Test viewing a file with line range."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(test_file),
                view_range=[2, 4],
                tool_context=tool_context,
            )
        )
        assert "line 2" in result
        assert "line 4" in result
        # Line 1 should not be shown (starts at line 2)
        assert "     1" not in result

    def test_view_with_range_end_minus_one(self, tool_context, tmp_path):
        """Test viewing with -1 as end of range."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(test_file),
                view_range=[2, -1],
                tool_context=tool_context,
            )
        )
        assert "line 2" in result
        assert "line 3" in result

    def test_view_directory(self, tool_context, tmp_path):
        """Test viewing a directory listing."""
        from strands.vended_tools.editor.editor import editor

        # Create some files
        (tmp_path / "file1.py").write_text("pass")
        (tmp_path / "file2.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(tmp_path),
                tool_context=tool_context,
            )
        )
        assert "file1.py" in result
        assert "file2.txt" in result
        assert "subdir/" in result

    def test_view_nonexistent(self, tool_context, tmp_path):
        """Test viewing a nonexistent file."""
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(tmp_path / "nonexistent.txt"),
                tool_context=tool_context,
            )
        )
        assert "does not exist" in result.lower()

    def test_view_invalid_range(self, tool_context, tmp_path):
        """Test viewing with invalid range."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.txt"
        test_file.write_text("line 1\nline 2\n")

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(test_file),
                view_range=[0, 2],
                tool_context=tool_context,
            )
        )
        assert "error" in result.lower()

    def test_create_file(self, tool_context, tmp_path):
        """Test creating a new file."""
        from strands.vended_tools.editor.editor import editor

        new_file = tmp_path / "new_file.py"

        result = run(
            editor.__wrapped__(
                command="create",
                path=str(new_file),
                file_text="print('hello')\n",
                tool_context=tool_context,
            )
        )
        assert "created" in result.lower()
        assert new_file.read_text() == "print('hello')\n"

    def test_create_existing_file(self, tool_context, tmp_path):
        """Test creating a file that already exists."""
        from strands.vended_tools.editor.editor import editor

        existing = tmp_path / "existing.py"
        existing.write_text("original")

        result = run(
            editor.__wrapped__(
                command="create",
                path=str(existing),
                file_text="new content",
                tool_context=tool_context,
            )
        )
        assert "already exists" in result.lower()
        assert existing.read_text() == "original"

    def test_create_missing_file_text(self, tool_context, tmp_path):
        """Test create without file_text."""
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="create",
                path=str(tmp_path / "new.py"),
                tool_context=tool_context,
            )
        )
        assert "file_text" in result.lower()

    def test_str_replace_unique(self, tool_context, tmp_path):
        """Test str_replace with a unique match."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'hello'\n")

        result = run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="return 'hello'",
                new_str="return 'world'",
                tool_context=tool_context,
            )
        )
        assert "edited" in result.lower()
        assert "return 'world'" in test_file.read_text()

    def test_str_replace_not_found(self, tool_context, tmp_path):
        """Test str_replace when old_str not found."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'hello'\n")

        result = run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="nonexistent string",
                new_str="replacement",
                tool_context=tool_context,
            )
        )
        assert "did not appear" in result.lower()

    def test_str_replace_multiple_occurrences(self, tool_context, tmp_path):
        """Test str_replace rejects multiple occurrences."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\ny = 1\nz = 1\n")

        result = run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="= 1",
                new_str="= 2",
                tool_context=tool_context,
            )
        )
        assert "multiple" in result.lower()
        # File should NOT be modified
        assert test_file.read_text() == "x = 1\ny = 1\nz = 1\n"

    def test_str_replace_deletion(self, tool_context, tmp_path):
        """Test str_replace with empty new_str (deletion)."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: remove this\ndef main():\n    pass\n")

        result = run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="# TODO: remove this\n",
                new_str="",
                tool_context=tool_context,
            )
        )
        assert "edited" in result.lower()
        assert "TODO" not in test_file.read_text()

    def test_insert(self, tool_context, tmp_path):
        """Test inserting text at a line."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 3\n")

        result = run(
            editor.__wrapped__(
                command="insert",
                path=str(test_file),
                insert_line=1,
                new_str="line 2",
                tool_context=tool_context,
            )
        )
        assert "edited" in result.lower()
        content = test_file.read_text()
        assert "line 1\nline 2\nline 3\n" == content

    def test_insert_at_beginning(self, tool_context, tmp_path):
        """Test inserting at the beginning of a file."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("line 2\nline 3\n")

        result = run(
            editor.__wrapped__(
                command="insert",
                path=str(test_file),
                insert_line=0,
                new_str="line 1",
                tool_context=tool_context,
            )
        )
        assert "edited" in result.lower()
        assert test_file.read_text().startswith("line 1\n")

    def test_insert_invalid_line(self, tool_context, tmp_path):
        """Test insert with invalid line number."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\n")

        result = run(
            editor.__wrapped__(
                command="insert",
                path=str(test_file),
                insert_line=999,
                new_str="new line",
                tool_context=tool_context,
            )
        )
        assert "error" in result.lower()

    def test_undo_edit(self, tool_context, tmp_path):
        """Test undo_edit reverting a str_replace."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "test.py"
        test_file.write_text("original content\n")

        # Make an edit
        run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="original content",
                new_str="modified content",
                tool_context=tool_context,
            )
        )
        assert "modified content" in test_file.read_text()

        # Undo
        result = run(
            editor.__wrapped__(
                command="undo_edit",
                path=str(test_file),
                tool_context=tool_context,
            )
        )
        assert "reverted" in result.lower()
        assert "original content" in test_file.read_text()

    def test_undo_no_history(self, tool_context, tmp_path):
        """Test undo_edit when no history exists."""
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="undo_edit",
                path=str(tmp_path / "nonexistent.py"),
                tool_context=tool_context,
            )
        )
        assert "no edit history" in result.lower()

    def test_relative_path_rejected(self, tool_context):
        """Test that relative paths are rejected."""
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="view",
                path="relative/path.py",
                tool_context=tool_context,
            )
        )
        assert "not an absolute path" in result.lower()

    def test_path_traversal_rejected(self, tool_context):
        """Test that path traversal is rejected."""
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="view",
                path="/tmp/../etc/passwd",
                tool_context=tool_context,
            )
        )
        assert "not allowed" in result.lower()

    def test_interrupt_confirmation(self, tool_context, mock_agent, tmp_path):
        """Test interrupt for write operations."""
        from strands.vended_tools.editor.editor import editor

        mock_agent.state.set("strands_editor_tool", {"require_confirmation": True})

        with patch.object(type(tool_context), "interrupt", return_value="approve"):
            result = run(
                editor.__wrapped__(
                    command="create",
                    path=str(tmp_path / "approved.py"),
                    file_text="approved content",
                    tool_context=tool_context,
                )
            )
            assert "created" in result.lower()

    def test_interrupt_denied(self, tool_context, mock_agent, tmp_path):
        """Test interrupt denial for write operations."""
        from strands.vended_tools.editor.editor import editor

        mock_agent.state.set("strands_editor_tool", {"require_confirmation": True})

        with patch.object(type(tool_context), "interrupt", return_value="deny"):
            result = run(
                editor.__wrapped__(
                    command="create",
                    path=str(tmp_path / "denied.py"),
                    file_text="content",
                    tool_context=tool_context,
                )
            )
            assert "not approved" in result.lower()

    def test_noop_sandbox(self, tool_context, mock_agent, tmp_path):
        """Test editor with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()
        from strands.vended_tools.editor.editor import editor

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(tmp_path / "test.py"),
                tool_context=tool_context,
            )
        )
        assert "error" in result.lower()

    def test_max_file_size(self, tool_context, mock_agent, tmp_path):
        """Test max file size configuration."""
        from strands.vended_tools.editor.editor import editor

        mock_agent.state.set("strands_editor_tool", {"max_file_size": 10})

        test_file = tmp_path / "large.txt"
        test_file.write_text("a" * 100)

        result = run(
            editor.__wrapped__(
                command="view",
                path=str(test_file),
                tool_context=tool_context,
            )
        )
        assert "exceeds" in result.lower()


# ============================================================
# Python REPL Tool Tests
# ============================================================


class TestPythonReplTool:
    """Tests for the python_repl vended tool."""

    def test_basic_code(self, tool_context):
        """Test basic Python code execution."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="print('hello from python')",
                tool_context=tool_context,
            )
        )
        assert "hello from python" in result

    def test_code_with_math(self, tool_context):
        """Test Python math execution."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="print(2 + 2)",
                tool_context=tool_context,
            )
        )
        assert "4" in result

    def test_code_with_error(self, tool_context):
        """Test Python code that raises an error."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="raise ValueError('test error')",
                tool_context=tool_context,
            )
        )
        assert "test error" in result or "ValueError" in result

    def test_code_with_import(self, tool_context):
        """Test Python code with imports."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="import json; print(json.dumps({'key': 'value'}))",
                tool_context=tool_context,
            )
        )
        assert "key" in result

    def test_timeout(self, tool_context):
        """Test code execution timeout."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="import time; time.sleep(10)",
                timeout=1,
                tool_context=tool_context,
            )
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    def test_config_timeout(self, tool_context, mock_agent):
        """Test timeout from config."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        mock_agent.state.set("strands_python_repl_tool", {"timeout": 1})
        result = run(
            python_repl.__wrapped__(
                code="import time; time.sleep(10)",
                tool_context=tool_context,
            )
        )
        assert "timed out" in result.lower() or "error" in result.lower()

    def test_reset(self, tool_context):
        """Test REPL reset."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="",
                reset=True,
                tool_context=tool_context,
            )
        )
        assert "reset" in result.lower()

    def test_multiline_code(self, tool_context):
        """Test multiline Python code."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        code = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print(fibonacci(10))
"""
        result = run(
            python_repl.__wrapped__(
                code=code,
                tool_context=tool_context,
            )
        )
        assert "55" in result

    def test_no_output(self, tool_context):
        """Test code with no output."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="x = 42",
                tool_context=tool_context,
            )
        )
        assert result == "(no output)"

    def test_noop_sandbox(self, tool_context, mock_agent):
        """Test python_repl with NoOpSandbox."""
        mock_agent.sandbox = NoOpSandbox()
        from strands.vended_tools.python_repl.python_repl import python_repl

        result = run(
            python_repl.__wrapped__(
                code="print('test')",
                tool_context=tool_context,
            )
        )
        assert "error" in result.lower()

    def test_interrupt_confirmation(self, tool_context, mock_agent):
        """Test interrupt confirmation for code execution."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        mock_agent.state.set("strands_python_repl_tool", {"require_confirmation": True})

        with patch.object(type(tool_context), "interrupt", return_value="approve"):
            result = run(
                python_repl.__wrapped__(
                    code="print('approved')",
                    tool_context=tool_context,
                )
            )
            assert "approved" in result

    def test_interrupt_denied(self, tool_context, mock_agent):
        """Test interrupt denial for code execution."""
        from strands.vended_tools.python_repl.python_repl import python_repl

        mock_agent.state.set("strands_python_repl_tool", {"require_confirmation": True})

        with patch.object(type(tool_context), "interrupt", return_value="deny"):
            result = run(
                python_repl.__wrapped__(
                    code="print('should not run')",
                    tool_context=tool_context,
                )
            )
            assert "not approved" in result.lower()


# ============================================================
# Integration Tests
# ============================================================


class TestVendedToolsImport:
    """Test that vended tools can be imported from the package."""

    def test_import_from_vended_tools(self):
        """Test importing from strands.vended_tools."""
        from strands.vended_tools import editor, python_repl, shell

        assert shell is not None
        assert editor is not None
        assert python_repl is not None

    def test_import_individual_tools(self):
        """Test importing individual tools."""
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

        # Verify tool specs have required fields
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


# ============================================================
# Configuration Persistence Tests
# ============================================================


class TestConfigPersistence:
    """Test that tool configuration persists via agent state."""

    def test_shell_config_persists(self, mock_agent, agent_state):
        """Test shell config is read from agent state."""
        agent_state.set("strands_shell_tool", {"timeout": 300, "require_confirmation": False})
        config = agent_state.get("strands_shell_tool")
        assert config["timeout"] == 300
        assert config["require_confirmation"] is False

    def test_editor_config_persists(self, mock_agent, agent_state):
        """Test editor config is read from agent state."""
        agent_state.set("strands_editor_tool", {"max_file_size": 2097152})
        config = agent_state.get("strands_editor_tool")
        assert config["max_file_size"] == 2097152

    def test_python_repl_config_persists(self, mock_agent, agent_state):
        """Test python_repl config is read from agent state."""
        agent_state.set("strands_python_repl_tool", {"timeout": 60})
        config = agent_state.get("strands_python_repl_tool")
        assert config["timeout"] == 60

    def test_undo_state_persists(self, tool_context, tmp_path):
        """Test that undo state is stored in agent state."""
        from strands.vended_tools.editor.editor import editor

        test_file = tmp_path / "undo_test.py"
        test_file.write_text("original\n")

        run(
            editor.__wrapped__(
                command="str_replace",
                path=str(test_file),
                old_str="original",
                new_str="modified",
                tool_context=tool_context,
            )
        )

        undo_state = tool_context.agent.state.get("_strands_editor_undo")
        assert undo_state is not None
        assert str(test_file) in undo_state
