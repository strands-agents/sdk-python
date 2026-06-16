"""Tests for the bash tools.

Mirrors ``strands-ts/src/vended-tools/bash/__tests__/bash.test.node.ts``:

- ``TestBash`` covers the host-session :data:`bash` tool (persistent shell,
  execute/restart, state persistence, stderr, timeout).
- ``TestMakeBash`` covers the stateless sandbox-routed :func:`make_bash` factory,
  exercised against a real ``NotASandboxLocalEnvironment``.

Tools are called directly (like a normal async function), mirroring TS's
``bash.invoke(...)``. These spawn a shell and require POSIX, so they are skipped
on Windows.
"""

import sys
from types import SimpleNamespace

import pytest

from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.types.tools import ToolContext
from strands.vended_tools.bash import (
    SANDBOX_BASH_DESCRIPTION,
    BashSessionError,
    BashTimeoutError,
    bash,
    make_bash,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


class _Agent:
    """Minimal weak-referenceable stand-in for an agent (host bash keys sessions by agent)."""


def _host_context() -> ToolContext:
    """ToolContext for the host bash tool; its agent only needs to be weak-referenceable."""
    return ToolContext(tool_use={"name": "bash", "toolUseId": "id", "input": {}}, agent=_Agent(), invocation_state={})


def _sandbox_context(sandbox: NotASandboxLocalEnvironment | None = None) -> ToolContext:
    """ToolContext whose agent exposes a sandbox, for the sandbox-routed tool."""
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(tool_use={"name": "bash", "toolUseId": "id", "input": {}}, agent=agent, invocation_state={})


class TestBash:
    """Tests for the host-session ``bash`` tool."""

    @pytest.mark.asyncio
    async def test_executes_and_returns_output(self):
        result = await bash(mode="execute", command='echo "Hello World"', tool_context=_host_context())
        assert "Hello World" in result["output"]
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_restart_returns_message(self):
        assert await bash(mode="restart", tool_context=_host_context()) == "Bash session restarted"

    @pytest.mark.asyncio
    async def test_rejects_invalid_mode(self):
        with pytest.raises(BashSessionError, match="Unknown mode"):
            await bash(mode="invalid", tool_context=_host_context())

    @pytest.mark.asyncio
    async def test_execute_requires_command(self):
        with pytest.raises(BashSessionError, match="command is required"):
            await bash(mode="execute", tool_context=_host_context())

    @pytest.mark.asyncio
    async def test_persists_environment_variables(self):
        ctx = _host_context()
        await bash(mode="execute", command='MY_VAR="persistent_value"', tool_context=ctx)
        result = await bash(mode="execute", command="echo $MY_VAR", tool_context=ctx)
        assert result["output"].strip() == "persistent_value"

    @pytest.mark.asyncio
    async def test_persists_working_directory(self):
        ctx = _host_context()
        await bash(mode="execute", command="cd /tmp", tool_context=ctx)
        result = await bash(mode="execute", command="pwd", tool_context=ctx)
        assert result["output"].strip().endswith("/tmp")

    @pytest.mark.asyncio
    async def test_restart_clears_state(self):
        ctx = _host_context()
        await bash(mode="execute", command='MY_VAR="exists"', tool_context=ctx)
        await bash(mode="restart", tool_context=ctx)
        result = await bash(mode="execute", command='echo "${MY_VAR:-empty}"', tool_context=ctx)
        assert result["output"].strip() == "empty"

    @pytest.mark.asyncio
    async def test_isolated_sessions_per_agent(self):
        ctx1, ctx2 = _host_context(), _host_context()
        await bash(mode="execute", command='AGENT_VAR="agent1"', tool_context=ctx1)
        result = await bash(mode="execute", command='echo "${AGENT_VAR:-empty}"', tool_context=ctx2)
        assert result["output"].strip() == "empty"

    @pytest.mark.asyncio
    async def test_restart_with_no_existing_session(self):
        ctx = _host_context()
        assert await bash(mode="restart", tool_context=ctx) == "Bash session restarted"
        result = await bash(mode="execute", command='echo "works"', tool_context=ctx)
        assert result["output"].strip() == "works"

    @pytest.mark.asyncio
    async def test_returns_empty_stderr_on_success(self):
        result = await bash(mode="execute", command='echo "success"', tool_context=_host_context())
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_captures_stderr(self):
        result = await bash(mode="execute", command="nonexistent_command_xyz", tool_context=_host_context())
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_separates_stdout_and_stderr(self):
        result = await bash(mode="execute", command="echo out; echo err >&2", tool_context=_host_context())
        assert result["output"].strip() == "out"
        assert result["error"].strip() == "err"

    @pytest.mark.asyncio
    async def test_empty_output(self):
        result = await bash(mode="execute", command="true", tool_context=_host_context())
        assert result["output"] == ""
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_long_output(self):
        result = await bash(
            mode="execute", command='for i in $(seq 1 100); do echo "Line $i"; done', tool_context=_host_context()
        )
        assert "Line 1" in result["output"]
        assert "Line 100" in result["output"]

    @pytest.mark.asyncio
    async def test_completes_before_timeout(self):
        result = await bash(mode="execute", command='echo "fast"', timeout=5, tool_context=_host_context())
        assert "fast" in result["output"]

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        with pytest.raises(BashTimeoutError):
            await bash(mode="execute", command="sleep 10", timeout=1, tool_context=_host_context())


class TestMakeBash:
    """Tests for the stateless sandbox-routed ``make_bash`` factory."""

    @pytest.fixture
    def sandbox_bash(self):
        return make_bash(NotASandboxLocalEnvironment())

    @pytest.mark.asyncio
    async def test_executes_command_via_sandbox(self, sandbox_bash):
        result = await sandbox_bash(command='echo "hello sandbox"', tool_context=_sandbox_context())
        assert "hello sandbox" in result["output"]
        assert result["error"] == ""

    @pytest.mark.asyncio
    async def test_captures_stderr_via_sandbox(self, sandbox_bash):
        result = await sandbox_bash(command='echo "oops" >&2', tool_context=_sandbox_context())
        assert "oops" in result["error"]

    @pytest.mark.asyncio
    async def test_does_not_persist_state_between_calls(self, sandbox_bash):
        # Each call runs in a fresh shell; an exported var must not survive to the next call.
        await sandbox_bash(command="export MY_VAR=hello", tool_context=_sandbox_context())
        result = await sandbox_bash(command='echo "${MY_VAR:-empty}"', tool_context=_sandbox_context())
        assert result["output"].strip() == "empty"

    @pytest.mark.asyncio
    async def test_respects_timeout(self, sandbox_bash):
        with pytest.raises(BashTimeoutError):
            await sandbox_bash(command="sleep 10", tool_context=_sandbox_context(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_reads_sandbox_from_agent_context_when_unbound(self):
        unbound = make_bash()
        result = await unbound(command="echo via-context", tool_context=_sandbox_context())
        assert "via-context" in result["output"]


class TestToolMetadata:
    """Tests for tool names, descriptions, and input schemas."""

    def test_bash_name(self):
        assert bash.tool_name == "bash"

    def test_bash_mode_is_enum(self):
        props = bash.tool_spec["inputSchema"]["json"]["properties"]
        assert props["mode"]["enum"] == ["execute", "restart"]
        assert "tool_context" not in props

    def test_make_bash_default_name(self):
        assert make_bash().tool_name == "bash"

    def test_make_bash_custom_name(self):
        assert make_bash(name="sandbox_bash").tool_name == "sandbox_bash"

    def test_make_bash_default_description(self):
        assert make_bash().tool_spec["description"] == SANDBOX_BASH_DESCRIPTION

    def test_make_bash_custom_description(self):
        assert make_bash(description="custom desc").tool_spec["description"] == "custom desc"

    def test_make_bash_schema_excludes_context(self):
        props = make_bash().tool_spec["inputSchema"]["json"]["properties"]
        assert "command" in props
        assert "timeout" in props
        assert "tool_context" not in props


class TestErrorClasses:
    """Tests for the bash error classes."""

    def test_bash_timeout_error_is_exception(self):
        assert isinstance(BashTimeoutError("x"), Exception)

    def test_bash_session_error_is_exception(self):
        assert isinstance(BashSessionError("x"), Exception)
