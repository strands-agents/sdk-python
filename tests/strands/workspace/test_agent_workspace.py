"""Tests for Agent + Workspace integration."""

from strands.agent.agent import Agent
from strands.workspace.base import Workspace
from strands.workspace.local import LocalWorkspace


class TestAgentWorkspaceIntegration:
    def test_agent_workspace_defaults_to_local_workspace(self) -> None:
        """Agent.workspace defaults to LocalWorkspace when not explicitly set."""
        agent = Agent(model="test")
        assert agent.workspace is not None
        assert isinstance(agent.workspace, LocalWorkspace)
        assert isinstance(agent.workspace, Workspace)

    def test_agent_workspace_accepts_local_workspace(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(model="test", workspace=workspace)
        assert agent.workspace is workspace
        assert isinstance(agent.workspace, Workspace)

    def test_agent_workspace_default_uses_cwd(self) -> None:
        """Default LocalWorkspace uses the current working directory."""
        import os

        agent = Agent(model="test")
        assert isinstance(agent.workspace, LocalWorkspace)
        assert agent.workspace.working_dir == os.getcwd()

    def test_agent_workspace_is_accessible(self, tmp_path: object) -> None:
        """Tools can access workspace via agent.workspace."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(model="test", workspace=workspace)
        assert agent.workspace.working_dir == str(tmp_path)


class TestAgentWorkspaceToolAccess:
    """Critical: Verify tools can access workspace through tool_context.agent.workspace."""

    def test_workspace_accessible_via_agent_attribute(self, tmp_path: object) -> None:
        """Simulates tool access pattern: tool_context.agent.workspace."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(model="test", workspace=workspace)

        # This is the access pattern tools use: tool_context.agent.workspace
        accessed_workspace = agent.workspace
        assert accessed_workspace is workspace
        assert accessed_workspace.working_dir == str(tmp_path)
        assert hasattr(accessed_workspace, "execute")
        assert hasattr(accessed_workspace, "read_file")
        assert hasattr(accessed_workspace, "write_file")
        assert hasattr(accessed_workspace, "remove_file")
        assert hasattr(accessed_workspace, "list_files")
        assert hasattr(accessed_workspace, "execute_code")

    def test_multiple_agents_share_workspace_correctly(self, tmp_path: object) -> None:
        """Two agents sharing a workspace should both access the same instance."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent1 = Agent(model="test", workspace=workspace)
        agent2 = Agent(model="test", workspace=workspace)
        assert agent1.workspace is agent2.workspace
        assert agent1.workspace.working_dir == agent2.workspace.working_dir

    def test_default_workspace_has_all_methods(self) -> None:
        """Default LocalWorkspace should have all abstract methods implemented."""
        agent = Agent(model="test")
        ws = agent.workspace
        # Verify all 6 abstract methods + 2 convenience methods exist
        for method in [
            "execute",
            "execute_code",
            "read_file",
            "write_file",
            "remove_file",
            "list_files",
            "read_text",
            "write_text",
        ]:
            assert callable(getattr(ws, method)), f"Missing method: {method}"
