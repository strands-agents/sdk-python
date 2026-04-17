"""Tests for Agent workspace integration."""

import pytest

from strands import Agent
from strands.workspace.base import ExecutionResult, Workspace
from strands.workspace.local import LocalWorkspace


class CustomWorkspace(Workspace):
    """Custom workspace implementing all 6 abstract methods for testing."""

    async def execute(self, command: str, timeout: int | None = None):  # type: ignore[override]
        yield ExecutionResult(exit_code=0, stdout="custom", stderr="")

    async def execute_code(self, code: str, language: str = "python", timeout: int | None = None):  # type: ignore[override]
        yield ExecutionResult(exit_code=0, stdout="custom code", stderr="")

    async def read_file(self, path: str) -> str:
        return "custom content"

    async def write_file(self, path: str, content: str) -> None:
        pass

    async def remove_file(self, path: str) -> None:
        pass

    async def list_files(self, path: str = ".") -> list[str]:
        return ["custom.txt"]


class TestAgentWorkspace:
    def test_default_workspace_is_local(self) -> None:
        agent = Agent()
        assert isinstance(agent.workspace, LocalWorkspace)

    def test_custom_workspace(self) -> None:
        custom = CustomWorkspace()
        agent = Agent(workspace=custom)
        assert agent.workspace is custom

    def test_explicit_local_workspace(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(workspace=workspace)
        assert agent.workspace is workspace
        assert agent.workspace.working_dir == str(tmp_path)

    def test_workspace_accessible_via_tool_context(self) -> None:
        """Verify workspace is accessible via agent.workspace (tool_context.agent.workspace path)."""
        custom = CustomWorkspace()
        agent = Agent(workspace=custom)
        assert agent.workspace is custom

    def test_multiple_agents_independent_workspaces(self) -> None:
        agent1 = Agent()
        agent2 = Agent()
        assert agent1.workspace is not agent2.workspace

    def test_agent_with_none_workspace_uses_default(self) -> None:
        agent = Agent(workspace=None)
        assert isinstance(agent.workspace, LocalWorkspace)

    def test_local_workspace_extends_workspace_not_shell_based(self) -> None:
        """Verify LocalWorkspace extends Workspace directly, not ShellBasedWorkspace."""
        from strands.workspace.shell_based import ShellBasedWorkspace

        agent = Agent()
        assert isinstance(agent.workspace, Workspace)
        assert not isinstance(agent.workspace, ShellBasedWorkspace)
