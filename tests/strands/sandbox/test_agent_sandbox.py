"""Tests for Agent sandbox integration."""

import unittest.mock

import pytest

from strands import Agent
from strands.sandbox.base import ExecutionResult, Sandbox
from strands.sandbox.local import LocalSandbox


class CustomSandbox(Sandbox):
    """Custom sandbox for testing sandbox parameter."""

    async def execute(self, command: str, timeout: int | None = None) -> ExecutionResult:
        return ExecutionResult(exit_code=0, stdout="custom", stderr="")


class TestAgentSandbox:
    def test_default_sandbox_is_local(self):
        agent = Agent()
        assert isinstance(agent.sandbox, LocalSandbox)

    def test_custom_sandbox(self):
        custom = CustomSandbox()
        agent = Agent(sandbox=custom)
        assert agent.sandbox is custom

    def test_explicit_local_sandbox(self, tmp_path):
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent = Agent(sandbox=sandbox)
        assert agent.sandbox is sandbox
        assert agent.sandbox.working_dir == str(tmp_path)

    def test_sandbox_accessible_via_tool_context(self):
        """Verify sandbox is accessible via agent.sandbox (tool_context.agent.sandbox path)."""
        custom = CustomSandbox()
        agent = Agent(sandbox=custom)
        # Tools access via tool_context.agent.sandbox
        assert agent.sandbox is custom

    def test_multiple_agents_independent_sandboxes(self):
        agent1 = Agent()
        agent2 = Agent()
        assert agent1.sandbox is not agent2.sandbox

    def test_agent_with_none_sandbox_uses_default(self):
        agent = Agent(sandbox=None)
        assert isinstance(agent.sandbox, LocalSandbox)
