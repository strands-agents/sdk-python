"""Tests for AgentCore deployment target."""

from unittest.mock import MagicMock, patch

import pytest

from strands.experimental.deploy import DeployConfig, DeployResult
from strands.experimental.deploy._agentcore import AgentCoreTarget
from strands.experimental.deploy._exceptions import DeployException, DeployTargetException
from strands.experimental.deploy._state import DeployState, StateManager


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test-agent"
    agent.system_prompt = "You are helpful."
    agent.model = MagicMock()
    agent.model.config = {"model_id": "us.anthropic.claude-sonnet-4-20250514"}
    return agent


@pytest.fixture
def config():
    return DeployConfig(
        target="agentcore",
        name="test-agent",
        region="us-west-2",
    )


@pytest.fixture
def state_manager(tmp_path):
    return StateManager(base_dir=str(tmp_path))


@pytest.fixture
def target():
    return AgentCoreTarget()


@pytest.fixture
def mock_runtime_class():
    """Create a mock Runtime class that returns a mock instance."""
    mock_instance = MagicMock()
    mock_instance.launch.return_value = MagicMock(
        agent_runtime_id="rt-abc123",
        agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/rt-abc123",
        agent_runtime_endpoint_arn="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/rt-abc123/endpoint/ep-1",
        role_arn="arn:aws:iam::123456789012:role/strands-test-agent-agentcore-role",
    )
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls, mock_instance


class TestAgentCoreValidate:
    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    @patch("boto3.client")
    def test_validate_succeeds_with_valid_setup(self, mock_client, mock_get_runtime, target, config):
        target.validate(config)
        mock_get_runtime.assert_called_once()
        mock_client.assert_called_with("sts")

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    @patch("boto3.client")
    def test_validate_fails_without_credentials(self, mock_client, mock_get_runtime, target, config):
        mock_client.return_value.get_caller_identity.side_effect = Exception("No credentials")
        with pytest.raises(DeployException, match="credentials"):
            target.validate(config)

    @patch("strands.experimental.deploy._agentcore._get_runtime_class", side_effect=DeployException("not installed"))
    def test_validate_fails_without_toolkit(self, mock_get_runtime, target, config):
        with pytest.raises(DeployException, match="not installed"):
            target.validate(config)


class TestAgentCoreDeploy:
    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_deploy_creates_new_runtime(self, mock_get_runtime, mock_runtime_class, target, mock_agent, config, state_manager):
        mock_cls, mock_instance = mock_runtime_class
        mock_get_runtime.return_value = mock_cls

        result = target.deploy(mock_agent, config, state_manager)

        # Verify result
        assert isinstance(result, DeployResult)
        assert result.target == "agentcore"
        assert result.name == "test-agent"
        assert result.region == "us-west-2"
        assert result.created is True
        assert result.agent_runtime_id == "rt-abc123"
        assert "rt-abc123" in result.agent_runtime_arn

        # Verify Runtime was configured and launched
        mock_instance.configure.assert_called_once()
        configure_kwargs = mock_instance.configure.call_args[1]
        assert configure_kwargs["agent_name"] == "strands_test-agent"
        assert configure_kwargs["deployment_type"] == "direct_code_deploy"
        assert configure_kwargs["region"] == "us-west-2"
        assert configure_kwargs["non_interactive"] is True
        mock_instance.launch.assert_called_once()

        # Verify state was saved
        saved = state_manager.load("test-agent")
        assert saved is not None
        assert saved["agent_runtime_id"] == "rt-abc123"

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_deploy_updates_existing_runtime(self, mock_get_runtime, mock_runtime_class, target, mock_agent, config, state_manager):
        # Pre-populate state with existing deployment
        state_manager.save(
            "test-agent",
            DeployState(
                target="agentcore",
                region="us-west-2",
                agent_runtime_id="rt-existing",
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt-existing",
                role_arn="arn:aws:iam::123:role/strands-test-agent-agentcore-role",
                agent_runtime_endpoint_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt-existing/endpoint/ep-1",
            ),
        )

        mock_cls, mock_instance = mock_runtime_class
        mock_get_runtime.return_value = mock_cls

        result = target.deploy(mock_agent, config, state_manager)

        assert result.created is False
        mock_instance.configure.assert_called_once()
        mock_instance.launch.assert_called_once()

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_deploy_passes_environment_variables(self, mock_get_runtime, mock_runtime_class, target, mock_agent, config, state_manager):
        mock_cls, mock_instance = mock_runtime_class
        mock_get_runtime.return_value = mock_cls
        config.environment_variables = {"API_KEY": "secret"}

        target.deploy(mock_agent, config, state_manager)

        configure_kwargs = mock_instance.configure.call_args[1]
        assert configure_kwargs["environment_variables"] == {"API_KEY": "secret"}

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_deploy_passes_description(self, mock_get_runtime, mock_runtime_class, target, mock_agent, config, state_manager):
        mock_cls, mock_instance = mock_runtime_class
        mock_get_runtime.return_value = mock_cls
        config.description = "A test agent"

        target.deploy(mock_agent, config, state_manager)

        configure_kwargs = mock_instance.configure.call_args[1]
        assert configure_kwargs["description"] == "A test agent"

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_deploy_raises_on_launch_failure(self, mock_get_runtime, target, mock_agent, config, state_manager):
        mock_instance = MagicMock()
        mock_instance.launch.side_effect = RuntimeError("Launch failed")
        mock_cls = MagicMock(return_value=mock_instance)
        mock_get_runtime.return_value = mock_cls

        with pytest.raises(DeployTargetException, match="Launch failed"):
            target.deploy(mock_agent, config, state_manager)


class TestAgentCoreResolveRegion:
    def test_uses_config_region(self, target, mock_agent, config):
        config.region = "eu-west-1"
        result = target._resolve_region(mock_agent, config)
        assert result == "eu-west-1"

    @patch("boto3.Session")
    def test_falls_back_to_boto3_session(self, mock_session, target, mock_agent, config):
        config.region = None
        mock_agent.model = MagicMock(spec=[])  # No config attr
        mock_session.return_value.region_name = "ap-southeast-1"

        result = target._resolve_region(mock_agent, config)
        assert result == "ap-southeast-1"


class TestAgentCoreDestroy:
    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_destroy_delegates_to_runtime(self, mock_get_runtime, target, state_manager):
        state_manager.save(
            "my-agent",
            DeployState(
                target="agentcore",
                region="us-west-2",
                agent_runtime_id="rt-123",
                role_arn="arn:aws:iam::123:role/strands-my-agent-agentcore-role",
            ),
        )

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        mock_get_runtime.return_value = mock_cls

        target.destroy("my-agent", state_manager)

        # Verify Runtime was configured and destroy was called
        mock_instance.configure.assert_called_once()
        configure_kwargs = mock_instance.configure.call_args[1]
        assert configure_kwargs["agent_name"] == "strands_my-agent"
        assert configure_kwargs["region"] == "us-west-2"
        mock_instance.destroy.assert_called_once()

        # Verify state was cleaned up
        assert state_manager.load("my-agent") is None

    @patch("strands.experimental.deploy._agentcore._get_runtime_class")
    def test_destroy_nonexistent_is_noop(self, mock_get_runtime, target, state_manager):
        target.destroy("nonexistent", state_manager)
        mock_get_runtime.assert_not_called()


class TestInjectUserAgent:
    def _make_fake_toolkit_modules(self):
        """Create fake toolkit module hierarchy for sys.modules patching."""
        import types

        pkg = types.ModuleType("bedrock_agentcore_starter_toolkit")
        services = types.ModuleType("bedrock_agentcore_starter_toolkit.services")
        runtime = types.ModuleType("bedrock_agentcore_starter_toolkit.services.runtime")
        runtime._get_user_agent = lambda: "agentcore-st/0.3.3"
        pkg.services = services
        services.runtime = runtime

        modules = {
            "bedrock_agentcore_starter_toolkit": pkg,
            "bedrock_agentcore_starter_toolkit.services": services,
            "bedrock_agentcore_starter_toolkit.services.runtime": runtime,
        }
        return modules, runtime

    def test_patches_toolkit_user_agent(self):
        from strands.experimental.deploy._agentcore import _inject_user_agent

        modules, runtime = self._make_fake_toolkit_modules()
        with patch.dict("sys.modules", modules):
            _inject_user_agent()
            assert "strands-agents-deploy" in runtime._get_user_agent()
            assert runtime._get_user_agent().startswith("agentcore-st/0.3.3")

    def test_idempotent(self):
        from strands.experimental.deploy._agentcore import _inject_user_agent

        modules, runtime = self._make_fake_toolkit_modules()
        with patch.dict("sys.modules", modules):
            _inject_user_agent()
            _inject_user_agent()  # Call twice
            ua = runtime._get_user_agent()
            assert ua.count("strands-agents-deploy") == 1

    def test_fails_silently_without_toolkit(self):
        from strands.experimental.deploy._agentcore import _inject_user_agent

        # Should not raise even if toolkit is not installed
        _inject_user_agent()
