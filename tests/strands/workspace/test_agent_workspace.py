"""Tests for Agent + Workspace integration."""

import pytest

from strands.agent.agent import Agent
from strands.workspace.base import Workspace
from strands.workspace.local import LocalWorkspace


class TestAgentWorkspaceIntegration:
    def test_agent_workspace_defaults_to_none(self) -> None:
        """Agent.workspace defaults to None when not explicitly set."""
        agent = Agent(model="test")
        assert agent.workspace is None

    def test_agent_workspace_accepts_local_workspace(self, tmp_path: object) -> None:
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(model="test", workspace=workspace)
        assert agent.workspace is workspace
        assert isinstance(agent.workspace, Workspace)

    def test_agent_workspace_accepts_none(self) -> None:
        agent = Agent(model="test", workspace=None)
        assert agent.workspace is None

    def test_agent_workspace_is_accessible(self, tmp_path: object) -> None:
        """Tools can access workspace via agent.workspace."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        agent = Agent(model="test", workspace=workspace)
        assert agent.workspace.working_dir == str(tmp_path)  # type: ignore[union-attr]
