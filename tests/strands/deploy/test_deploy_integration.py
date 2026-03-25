"""Tests for Agent.deploy() method integration."""

from unittest.mock import MagicMock, patch

import pytest

from strands.deploy import DeployConfig, DeployResult


class TestAgentDeployMethod:
    def test_deploy_method_exists_on_agent(self):
        from strands import Agent

        assert hasattr(Agent, "deploy")

    @patch("strands.deploy.deploy")
    def test_deploy_delegates_to_deploy_module(self, mock_deploy):
        from strands import Agent

        mock_deploy.return_value = DeployResult(
            target="agentcore",
            name="test-agent",
            region="us-west-2",
            created=True,
        )

        agent = MagicMock(spec=Agent)
        agent.name = "Test Agent"
        agent.deploy = Agent.deploy.__get__(agent, Agent)

        result = agent.deploy(target="agentcore", name="test-agent", region="us-west-2")

        assert isinstance(result, DeployResult)
        assert result.target == "agentcore"
        mock_deploy.assert_called_once()

        # Verify the config was constructed correctly
        call_args = mock_deploy.call_args
        config = call_args[0][1]
        assert isinstance(config, DeployConfig)
        assert config.target == "agentcore"
        assert config.name == "test-agent"
        assert config.region == "us-west-2"

    @patch("strands.deploy.deploy")
    def test_deploy_sanitizes_agent_name(self, mock_deploy):
        from strands import Agent

        mock_deploy.return_value = DeployResult(
            target="agentcore", name="my-special-agent", region="us-east-1", created=True
        )

        agent = MagicMock(spec=Agent)
        agent.name = "My Special Agent!!!"
        agent.deploy = Agent.deploy.__get__(agent, Agent)

        agent.deploy(target="agentcore")

        config = mock_deploy.call_args[0][1]
        # Should be sanitized: lowercase, special chars replaced with hyphens
        assert config.name == "my-special-agent"

    @patch("strands.deploy.deploy")
    def test_deploy_passes_environment_variables(self, mock_deploy):
        from strands import Agent

        mock_deploy.return_value = DeployResult(
            target="agentcore", name="test", region="us-east-1", created=True
        )

        agent = MagicMock(spec=Agent)
        agent.name = "test"
        agent.deploy = Agent.deploy.__get__(agent, Agent)

        env_vars = {"API_KEY": "secret", "LOG_LEVEL": "DEBUG"}
        agent.deploy(target="agentcore", environment_variables=env_vars)

        config = mock_deploy.call_args[0][1]
        assert config.environment_variables == env_vars


class TestDeployFunction:
    def test_unknown_target_raises(self):
        from strands.deploy import deploy
        from strands.deploy._exceptions import DeployException

        agent = MagicMock()
        config = DeployConfig.__new__(DeployConfig)
        config.target = "unknown_target"
        config.name = "test"

        with pytest.raises(DeployException, match="Unknown deploy target"):
            deploy(agent, config)
