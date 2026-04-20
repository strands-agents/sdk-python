"""Tests for Agent + Sandbox integration."""

from strands.agent.agent import Agent
from strands.sandbox.base import Sandbox
from strands.sandbox.local import LocalSandbox


class TestAgentSandboxIntegration:
    def test_agent_sandbox_defaults_to_local_sandbox(self) -> None:
        """Agent.sandbox defaults to LocalSandbox when not explicitly set."""
        agent = Agent(model="test")
        assert agent.sandbox is not None
        assert isinstance(agent.sandbox, LocalSandbox)
        assert isinstance(agent.sandbox, Sandbox)

    def test_agent_sandbox_accepts_local_sandbox(self, tmp_path: object) -> None:
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent = Agent(model="test", sandbox=sandbox)
        assert agent.sandbox is sandbox
        assert isinstance(agent.sandbox, Sandbox)

    def test_agent_sandbox_default_uses_cwd(self) -> None:
        """Default LocalSandbox uses the current working directory."""
        import os

        agent = Agent(model="test")
        assert isinstance(agent.sandbox, LocalSandbox)
        assert agent.sandbox.working_dir == os.getcwd()

    def test_agent_sandbox_is_accessible(self, tmp_path: object) -> None:
        """Tools can access sandbox via agent.sandbox."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent = Agent(model="test", sandbox=sandbox)
        assert agent.sandbox.working_dir == str(tmp_path)


class TestAgentSandboxToolAccess:
    """Critical: Verify tools can access sandbox through tool_context.agent.sandbox."""

    def test_sandbox_accessible_via_agent_attribute(self, tmp_path: object) -> None:
        """Simulates tool access pattern: tool_context.agent.sandbox."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent = Agent(model="test", sandbox=sandbox)

        # This is the access pattern tools use: tool_context.agent.sandbox
        accessed_sandbox = agent.sandbox
        assert accessed_sandbox is sandbox
        assert accessed_sandbox.working_dir == str(tmp_path)
        assert hasattr(accessed_sandbox, "execute")
        assert hasattr(accessed_sandbox, "read_file")
        assert hasattr(accessed_sandbox, "write_file")
        assert hasattr(accessed_sandbox, "remove_file")
        assert hasattr(accessed_sandbox, "list_files")
        assert hasattr(accessed_sandbox, "execute_code")

    def test_multiple_agents_share_sandbox_correctly(self, tmp_path: object) -> None:
        """Two agents sharing a sandbox should both access the same instance."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        agent1 = Agent(model="test", sandbox=sandbox)
        agent2 = Agent(model="test", sandbox=sandbox)
        assert agent1.sandbox is agent2.sandbox
        assert agent1.sandbox.working_dir == agent2.sandbox.working_dir

    def test_default_sandbox_has_all_methods(self) -> None:
        """Default LocalSandbox should have all abstract methods implemented."""
        agent = Agent(model="test")
        ws = agent.sandbox
        # Verify all 6 abstract methods + 2 convenience methods exist
        for method in [
            "execute",
            "execute_streaming",
            "execute_code",
            "execute_code_streaming",
            "read_file",
            "write_file",
            "remove_file",
            "list_files",
            "read_text",
            "write_text",
        ]:
            assert callable(getattr(ws, method)), f"Missing method: {method}"
