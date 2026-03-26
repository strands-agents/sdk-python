"""Tests for strands.experimental.deploy() function."""

from unittest.mock import MagicMock, patch

import pytest

from strands.experimental.deploy import DeployResult


class TestDeployFunction:
    @patch("strands.experimental.deploy._agentcore.AgentCoreTarget")
    def test_deploy_creates_config_and_delegates(self, mock_target_cls):
        from strands.experimental.deploy import deploy

        mock_target = MagicMock()
        mock_target.deploy.return_value = DeployResult(
            target="agentcore",
            name="test-agent",
            region="us-west-2",
            created=True,
        )
        mock_target_cls.return_value = mock_target

        agent = MagicMock()
        agent.name = "Test Agent"

        result = deploy(agent, target="agentcore", name="test-agent", region="us-west-2")

        assert isinstance(result, DeployResult)
        assert result.target == "agentcore"
        mock_target.validate.assert_called_once()
        mock_target.deploy.assert_called_once()

        # Verify the config was constructed correctly
        call_args = mock_target.deploy.call_args
        config = call_args[0][1]
        assert config.target == "agentcore"
        assert config.name == "test-agent"
        assert config.region == "us-west-2"

    @patch("strands.experimental.deploy._agentcore.AgentCoreTarget")
    def test_deploy_sanitizes_agent_name(self, mock_target_cls):
        from strands.experimental.deploy import deploy

        mock_target = MagicMock()
        mock_target.deploy.return_value = DeployResult(
            target="agentcore", name="my_special_agent", region="us-east-1", created=True
        )
        mock_target_cls.return_value = mock_target

        agent = MagicMock()
        agent.name = "My Special Agent!!!"

        deploy(agent, target="agentcore")

        config = mock_target.deploy.call_args[0][1]
        # Should be sanitized: lowercase, special chars replaced with underscores, stripped
        assert config.name == "my_special_agent"

    @patch("strands.experimental.deploy._agentcore.AgentCoreTarget")
    def test_deploy_passes_environment_variables(self, mock_target_cls):
        from strands.experimental.deploy import deploy

        mock_target = MagicMock()
        mock_target.deploy.return_value = DeployResult(
            target="agentcore", name="test", region="us-east-1", created=True
        )
        mock_target_cls.return_value = mock_target

        agent = MagicMock()
        agent.name = "test"

        env_vars = {"API_KEY": "secret", "LOG_LEVEL": "DEBUG"}
        deploy(agent, target="agentcore", environment_variables=env_vars)

        config = mock_target.deploy.call_args[0][1]
        assert config.environment_variables == env_vars

    def test_unknown_target_raises(self):
        from strands.experimental.deploy import deploy
        from strands.experimental.deploy._exceptions import DeployException

        agent = MagicMock()
        agent.name = "test"

        with pytest.raises(DeployException, match="Unknown deploy target"):
            deploy(agent, target="unknown_target")

    def test_agent_no_longer_has_deploy_method(self):
        from strands import Agent

        assert not hasattr(Agent, "deploy")

    def test_importable_from_experimental(self):
        from strands.experimental import deploy

        assert callable(deploy)
