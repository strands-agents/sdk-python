"""Tests for the bash tool.

The bash tool routes commands through the agent's sandbox (or a bound one).
Each call runs in a fresh shell; state does not persist across calls. These
spawn ``sh`` and require POSIX, so they are skipped on Windows.
"""

import sys
from types import SimpleNamespace

import pytest

from strands.sandbox.errors import SandboxTimeoutError
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.types.tools import ToolContext
from strands.vended_tools.bash import bash, make_bash
from strands.vended_tools.bash.types import SANDBOX_BASH_DESCRIPTION

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


def _tool_context(sandbox: NotASandboxLocalEnvironment | None = None) -> ToolContext:
    """Build a ToolContext whose agent exposes a sandbox (or a fresh one)."""
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(tool_use={"name": "bash", "toolUseId": "id", "input": {}}, agent=agent, invocation_state={})


class TestMakeBash:
    """Tests for a bash tool with a sandbox bound at creation."""

    @pytest.fixture
    def sandbox_bash(self):
        return make_bash(sandbox=NotASandboxLocalEnvironment())

    @pytest.mark.asyncio
    async def test_executes_command_via_sandbox(self, sandbox_bash):
        result = await sandbox_bash(command='echo "hello sandbox"', tool_context=_tool_context())
        assert "hello sandbox" in result["output"]
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_captures_stderr_via_sandbox(self, sandbox_bash):
        result = await sandbox_bash(command='echo "oops" >&2', tool_context=_tool_context())
        assert "oops" in result["error"]

    @pytest.mark.asyncio
    async def test_does_not_persist_state_between_calls(self, sandbox_bash):
        await sandbox_bash(command="export MY_VAR=hello", tool_context=_tool_context())
        result = await sandbox_bash(command='echo "${MY_VAR:-empty}"', tool_context=_tool_context())
        assert result["output"].strip() == "empty"

    @pytest.mark.asyncio
    async def test_respects_timeout(self, sandbox_bash):
        with pytest.raises(SandboxTimeoutError):
            await sandbox_bash(command="sleep 10", tool_context=_tool_context(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wraps_sandbox_error_as_runtime_error(self):
        class _BoomSandbox(NotASandboxLocalEnvironment):
            async def execute(self, *args, **kwargs):
                raise ValueError("boom")

        t = make_bash(sandbox=_BoomSandbox())
        with pytest.raises(RuntimeError, match="boom"):
            await t(command="echo hi", tool_context=_tool_context())


class TestDefaultBash:
    """Tests for the default unbound ``bash`` instance (reads sandbox from agent context)."""

    @pytest.mark.asyncio
    async def test_reads_sandbox_from_agent_context(self):
        result = await bash(command="echo via-context", tool_context=_tool_context())
        assert "via-context" in result["output"]


class TestToolMetadata:
    """Tests for tool names, descriptions, and input schemas."""

    def test_default_name(self):
        assert bash.tool_name == "bash"

    def test_custom_name(self):
        assert make_bash(name="sandbox_bash").tool_name == "sandbox_bash"

    def test_default_description(self):
        assert make_bash().tool_spec["description"] == SANDBOX_BASH_DESCRIPTION

    def test_custom_description(self):
        assert make_bash(description="custom desc").tool_spec["description"] == "custom desc"

    def test_schema_excludes_context(self):
        props = bash.tool_spec["inputSchema"]["json"]["properties"]
        assert "command" in props
        assert "timeout" in props
        assert "tool_context" not in props
