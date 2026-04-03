"""Tests for Agent sandbox integration."""

import pytest

from strands import Agent
from strands.sandbox.base import ExecutionResult, Sandbox, ShellBasedSandbox
from strands.sandbox.local import LocalSandbox


class CustomSandbox(Sandbox):
    """Custom sandbox implementing all 5 abstract methods for testing."""

    async def execute(self, command: str, timeout: int | None = None):  # type: ignore[override]
        yield ExecutionResult(exit_code=0, stdout="custom", stderr="")

    async def execute_code(self, code: str, language: str = "python", timeout: int | None = None):  # type: ignore[override]
        yield ExecutionResult(exit_code=0, stdout="custom code", stderr="")

    async def read_file(self, path: str) -> str:
        return "custom content"

    async def write_file(self, path: str, content: str) -> None:
        pass

    async def list_files(self, path: str = ".") -> list[str]:
        return ["custom.txt"]


class TestAgentSandbox:
    def test_default_sandbox_is_local(self) -> None:
        agent = Agent()
        assert isinstance(agent.sandbox, LocalSandbox)

    def test_custom_sandbox(self) -> None:
        custom = CustomSandbox()
        agent = Agent(sandbox=custom)
        assert agent.sandbox is custom

    def test_explicit_local_sandbox(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent = Agent(sandbox=sandbox)
        assert agent.sandbox is sandbox
        assert agent.sandbox.working_dir == str(tmp_path)

    def test_sandbox_accessible_via_tool_context(self) -> None:
        """Verify sandbox is accessible via agent.sandbox (tool_context.agent.sandbox path)."""
        custom = CustomSandbox()
        agent = Agent(sandbox=custom)
        # Tools access via tool_context.agent.sandbox
        assert agent.sandbox is custom

    def test_multiple_agents_independent_sandboxes(self) -> None:
        agent1 = Agent()
        agent2 = Agent()
        assert agent1.sandbox is not agent2.sandbox

    def test_agent_with_none_sandbox_uses_default(self) -> None:
        agent = Agent(sandbox=None)
        assert isinstance(agent.sandbox, LocalSandbox)

    def test_local_sandbox_is_shell_based(self) -> None:
        """Verify LocalSandbox inherits from ShellBasedSandbox."""
        agent = Agent()
        assert isinstance(agent.sandbox, ShellBasedSandbox)
        assert isinstance(agent.sandbox, Sandbox)
