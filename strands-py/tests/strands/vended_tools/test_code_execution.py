"""Tests for the code_execution tool.

The tool shims :meth:`~strands.sandbox.base.Sandbox.execute_code`. Security tests
run against a fake sandbox to exercise the tool boundary; happy-path tests run
against a :class:`~strands.sandbox.posix_shell.PosixShellSandbox` subclass that
executes ``python3`` on the host, mirroring the sandbox tests' fixture. These
spawn ``sh``/``python3`` and require POSIX, so they are skipped on Windows.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest

from strands.sandbox.base import Sandbox
from strands.sandbox.errors import SandboxTimeoutError
from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment
from strands.sandbox.posix_shell import PosixShellSandbox, build_shell_env_prefix
from strands.sandbox.stream_process import _stream_process
from strands.sandbox.types import ExecutionResult, StreamChunk
from strands.types.tools import ToolContext
from strands.vended_tools.code_execution import code_execution, make_code_execution
from strands.vended_tools.code_execution.types import (
    CODE_EXECUTION_DESCRIPTION,
    DEFAULT_MAX_CODE_BYTES,
    TRUNCATION_MARKER,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell required")


class _ShellTestSandbox(PosixShellSandbox):
    """Concrete sandbox that runs commands via ``sh -c`` in a working directory."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        target_cwd = cwd if cwd is not None else self.working_dir
        env_prefix = build_shell_env_prefix(env)
        full_command = f"cd {shlex.quote(target_cwd)} && {env_prefix}{command}"
        async for chunk in _stream_process("sh", ["-c", full_command], timeout=timeout):
            yield chunk


def _tool_context(sandbox: Sandbox | None = None) -> ToolContext:
    """Build a ToolContext whose agent exposes the given sandbox (or the host default)."""
    agent = SimpleNamespace(sandbox=sandbox or NotASandboxLocalEnvironment())
    return ToolContext(
        tool_use={"name": "code_execution", "toolUseId": "id", "input": {}},
        agent=agent,
        invocation_state={},
    )


@pytest.fixture
def isolating_sandbox(tmp_path) -> _ShellTestSandbox:
    """A sandbox subclass that is *not* NotASandboxLocalEnvironment.

    The tool treats NotASandboxLocalEnvironment as "no isolation" and refuses to
    execute; this fixture stands in for a real Docker/SSH sandbox for happy-path
    tests without requiring one to be provisioned.
    """
    return _ShellTestSandbox(str(tmp_path))


# ---- Security surface: refuses when no isolating sandbox ----


class TestRefusalWithoutSandbox:
    """The tool must refuse when the agent has no isolating sandbox configured."""

    @pytest.mark.asyncio
    async def test_refuses_when_agent_sandbox_is_host_default(self):
        # The default agent sandbox is NotASandboxLocalEnvironment (no isolation).
        with pytest.raises(RuntimeError, match="requires an isolating sandbox"):
            await code_execution(code="print('nope')", tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_refuses_even_when_bound_at_creation(self):
        # If someone binds NotASandboxLocalEnvironment at creation, still refuse.
        t = make_code_execution(sandbox=NotASandboxLocalEnvironment())
        with pytest.raises(RuntimeError, match="requires an isolating sandbox"):
            await t(code="print('nope')", tool_context=_tool_context())


# ---- Security surface: oversized inputs ----


class TestInputCaps:
    """Oversized code is rejected at the tool boundary before touching the sandbox."""

    @pytest.mark.asyncio
    async def test_rejects_oversized_code(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox, max_code_bytes=100)
        oversized = "x" * 200
        with pytest.raises(ValueError, match="exceeds maximum"):
            await t(code=oversized, tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_default_cap_is_reasonable(self, isolating_sandbox):
        # Under the cap: runs. Over the cap: refuses. This locks in the default.
        t = make_code_execution(sandbox=isolating_sandbox)
        code = "a" * (DEFAULT_MAX_CODE_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            await t(code=code, tool_context=_tool_context())

    @pytest.mark.asyncio
    async def test_truncates_stdout_over_limit(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox, max_output_bytes=32)
        # Emit more than 32 bytes of output.
        result = await t(code="print('x' * 200)", tool_context=_tool_context())
        assert result["stdout"].endswith(TRUNCATION_MARKER)
        assert len(result["stdout"].encode("utf-8")) <= 32 + len(TRUNCATION_MARKER.encode("utf-8"))

    def test_factory_rejects_nonpositive_caps(self):
        with pytest.raises(ValueError):
            make_code_execution(max_code_bytes=0)
        with pytest.raises(ValueError):
            make_code_execution(max_output_bytes=-1)
        with pytest.raises(ValueError):
            make_code_execution(default_timeout=0)

    def test_factory_rejects_non_finite_default_timeout(self):
        # Without the finiteness check, ``nan <= 0`` is false and the sandbox
        # would see a non-finite timeout.
        with pytest.raises(ValueError):
            make_code_execution(default_timeout=float("nan"))
        with pytest.raises(ValueError):
            make_code_execution(default_timeout=float("inf"))

    def test_factory_rejects_boolean_caps(self):
        # ``bool`` is a subclass of ``int``; catch it explicitly so a stray
        # ``True`` cannot be smuggled through as a "positive integer".
        with pytest.raises(ValueError):
            make_code_execution(max_code_bytes=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            make_code_execution(max_output_bytes=False)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_rejects_nonpositive_timeout(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        with pytest.raises(ValueError, match="timeout must be"):
            await t(code="print(1)", tool_context=_tool_context(), timeout=0)

    @pytest.mark.asyncio
    async def test_rejects_non_finite_timeout(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        with pytest.raises(ValueError, match="finite"):
            await t(code="print(1)", tool_context=_tool_context(), timeout=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            await t(code="print(1)", tool_context=_tool_context(), timeout=float("inf"))


# ---- Happy path ----


class TestHappyPath:
    """The tool executes code through the sandbox and returns the expected shape."""

    @pytest.mark.asyncio
    async def test_returns_stdout(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        result = await t(code="print(2 + 2)", tool_context=_tool_context())
        assert result["stdout"].strip() == "4"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert isinstance(result["elapsed_ms"], int)
        assert result["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_returns_stderr(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        result = await t(code="import sys; sys.stderr.write('oops')", tool_context=_tool_context())
        assert "oops" in result["stderr"]

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_on_failure(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        result = await t(code="import sys; sys.exit(3)", tool_context=_tool_context())
        assert result["exit_code"] == 3

    @pytest.mark.asyncio
    async def test_syntax_error_surfaces_as_nonzero_exit(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        result = await t(code="def : bad(", tool_context=_tool_context())
        assert result["exit_code"] != 0
        assert result["stderr"] != ""

    @pytest.mark.asyncio
    async def test_reads_sandbox_from_agent_context(self, isolating_sandbox):
        # Default (unbound) instance falls through to context.agent.sandbox.
        result = await code_execution(
            code="print('via-context')",
            tool_context=_tool_context(isolating_sandbox),
        )
        assert result["stdout"].strip() == "via-context"


# ---- Timeout ----


class TestTimeout:
    """Timeout is passed through to the sandbox and surfaces as SandboxTimeoutError."""

    @pytest.mark.asyncio
    async def test_timeout_on_runaway(self, isolating_sandbox):
        t = make_code_execution(sandbox=isolating_sandbox)
        with pytest.raises(SandboxTimeoutError):
            await t(
                code="import time; time.sleep(10)",
                tool_context=_tool_context(),
                timeout=0.2,
            )


# ---- Error passthrough ----


class TestSandboxErrorPassthrough:
    """Non-timeout sandbox errors surface as RuntimeError."""

    @pytest.mark.asyncio
    async def test_wraps_arbitrary_sandbox_error(self):
        class _BoomSandbox(_ShellTestSandbox):
            async def execute_code(self, *args, **kwargs):  # type: ignore[override]
                raise ValueError("boom")

        t = make_code_execution(sandbox=_BoomSandbox("/tmp"))
        with pytest.raises(RuntimeError, match="boom"):
            await t(code="print(1)", tool_context=_tool_context())


# ---- Tool metadata ----


class TestToolMetadata:
    """Tool names, descriptions, and input schemas are correct."""

    def test_default_name(self):
        assert code_execution.tool_name == "code_execution"

    def test_custom_name(self):
        assert make_code_execution(name="sandbox_code").tool_name == "sandbox_code"

    def test_default_description(self):
        assert make_code_execution().tool_spec["description"] == CODE_EXECUTION_DESCRIPTION

    def test_custom_description(self):
        assert make_code_execution(description="custom desc").tool_spec["description"] == "custom desc"

    def test_schema_excludes_context(self):
        props = code_execution.tool_spec["inputSchema"]["json"]["properties"]
        assert "code" in props
        assert "timeout" in props
        assert "tool_context" not in props

    def test_timeout_description_reflects_default(self):
        # The model-facing docstring should advertise the factory's configured
        # default rather than a hardcoded literal.
        t = make_code_execution(default_timeout=15)
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        assert "15" in props["timeout"]["description"]
