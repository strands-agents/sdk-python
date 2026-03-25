"""Tests for AgentCore deployment target."""

from unittest.mock import MagicMock, patch

import pytest

from strands.deploy import DeployConfig, DeployResult
from strands.deploy._agentcore import AgentCoreTarget
from strands.deploy._exceptions import DeployException, DeployTargetException
from strands.deploy._state import DeployState, StateManager


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
        auth="public",
        region="us-west-2",
    )


@pytest.fixture
def state_manager(tmp_path):
    return StateManager(base_dir=str(tmp_path))


@pytest.fixture
def target():
    return AgentCoreTarget()


class TestAgentCoreValidate:
    @patch("strands.deploy._agentcore.boto3")
    def test_validate_succeeds_with_valid_setup(self, mock_boto3, target, config):
        target.validate(config)
        mock_boto3.client.assert_any_call("bedrock-agentcore-control", region_name="us-east-1")
        mock_boto3.client.assert_any_call("sts")

    @patch("strands.deploy._agentcore.boto3")
    def test_validate_fails_without_credentials(self, mock_boto3, target, config):
        mock_boto3.client.return_value.get_caller_identity.side_effect = Exception("No credentials")
        with pytest.raises(DeployException, match="credentials"):
            target.validate(config)


class TestAgentCoreDeploy:
    @patch("strands.deploy._agentcore.time.sleep")
    @patch("strands.deploy._agentcore.boto3")
    @patch("strands.deploy._packaging.boto3")
    def test_deploy_creates_new_runtime(
        self, mock_packaging_boto3, mock_boto3, mock_sleep, target, mock_agent, config, state_manager
    ):
        # Setup mocks
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_iam = MagicMock()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/strands-test-agent-agentcore-role"}
        }

        mock_control = MagicMock()
        mock_control.create_agent_runtime.return_value = {
            "agentRuntimeId": "rt-abc123",
            "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/rt-abc123",
            "agentRuntimeVersion": "1",
            "status": "CREATING",
        }
        mock_control.get_agent_runtime.return_value = {
            "status": "READY",
        }
        mock_control.create_agent_runtime_endpoint.return_value = {
            "agentRuntimeEndpointArn": (
                "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/rt-abc123/endpoint/ep-1"
            ),
        }

        mock_s3 = MagicMock()

        def client_factory(service, **kwargs):
            clients = {
                "sts": mock_sts,
                "iam": mock_iam,
                "bedrock-agentcore-control": mock_control,
                "s3": mock_s3,
            }
            return clients.get(service, MagicMock())

        mock_boto3.client.side_effect = client_factory
        mock_boto3.Session.return_value.region_name = "us-west-2"
        mock_packaging_boto3.client.side_effect = client_factory

        result = target.deploy(mock_agent, config, state_manager)

        # Verify result
        assert isinstance(result, DeployResult)
        assert result.target == "agentcore"
        assert result.name == "test-agent"
        assert result.region == "us-west-2"
        assert result.created is True
        assert result.agent_runtime_id == "rt-abc123"
        assert "rt-abc123" in result.agent_runtime_arn

        # Verify IAM role was created
        mock_iam.create_role.assert_called_once()
        create_role_kwargs = mock_iam.create_role.call_args[1]
        assert create_role_kwargs["RoleName"] == "strands-test-agent-agentcore-role"

        # Verify runtime was created (not updated)
        mock_control.create_agent_runtime.assert_called_once()
        create_kwargs = mock_control.create_agent_runtime.call_args[1]
        assert create_kwargs["agentRuntimeName"] == "strands-test-agent"
        assert "codeConfiguration" in create_kwargs["agentRuntimeArtifact"]
        assert create_kwargs["networkConfiguration"]["networkMode"] == "PUBLIC"

        # Verify endpoint was created
        mock_control.create_agent_runtime_endpoint.assert_called_once()

        # Verify state was saved
        saved = state_manager.load("test-agent")
        assert saved is not None
        assert saved["agent_runtime_id"] == "rt-abc123"

    @patch("strands.deploy._agentcore.time.sleep")
    @patch("strands.deploy._agentcore.boto3")
    @patch("strands.deploy._packaging.boto3")
    def test_deploy_updates_existing_runtime(
        self, mock_packaging_boto3, mock_boto3, mock_sleep, target, mock_agent, config, state_manager
    ):
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

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_control = MagicMock()
        mock_control.update_agent_runtime.return_value = {
            "agentRuntimeId": "rt-existing",
            "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt-existing",
            "agentRuntimeVersion": "2",
            "status": "UPDATING",
        }
        mock_control.get_agent_runtime.return_value = {"status": "READY"}

        mock_s3 = MagicMock()

        def client_factory(service, **kwargs):
            return {"sts": mock_sts, "bedrock-agentcore-control": mock_control, "s3": mock_s3}.get(
                service, MagicMock()
            )

        mock_boto3.client.side_effect = client_factory
        mock_packaging_boto3.client.side_effect = client_factory

        result = target.deploy(mock_agent, config, state_manager)

        assert result.created is False
        assert result.agent_runtime_id == "rt-existing"

        # Should update, not create
        mock_control.update_agent_runtime.assert_called_once()
        mock_control.create_agent_runtime.assert_not_called()

        # Should NOT create a new endpoint
        mock_control.create_agent_runtime_endpoint.assert_not_called()

    @patch("strands.deploy._agentcore.time.sleep")
    @patch("strands.deploy._agentcore.boto3")
    @patch("strands.deploy._packaging.boto3")
    def test_deploy_raises_on_runtime_failure(
        self, mock_packaging_boto3, mock_boto3, mock_sleep, target, mock_agent, config, state_manager
    ):
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_iam = MagicMock()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123:role/test-role"}
        }

        mock_control = MagicMock()
        mock_control.create_agent_runtime.return_value = {
            "agentRuntimeId": "rt-fail",
            "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/rt-fail",
            "status": "CREATING",
        }
        mock_control.get_agent_runtime.return_value = {
            "status": "CREATE_FAILED",
            "failureReason": "Invalid configuration",
        }

        mock_s3 = MagicMock()

        def client_factory(service, **kwargs):
            return {"sts": mock_sts, "iam": mock_iam, "bedrock-agentcore-control": mock_control, "s3": mock_s3}.get(
                service, MagicMock()
            )

        mock_boto3.client.side_effect = client_factory
        mock_packaging_boto3.client.side_effect = client_factory

        with pytest.raises(DeployTargetException, match="CREATE_FAILED"):
            target.deploy(mock_agent, config, state_manager)


class TestAgentCoreResolveRegion:
    def test_uses_config_region(self, target, mock_agent, config):
        config.region = "eu-west-1"
        result = target._resolve_region(mock_agent, config)
        assert result == "eu-west-1"

    @patch("strands.deploy._agentcore.boto3")
    def test_falls_back_to_boto3_session(self, mock_boto3, target, mock_agent, config):
        config.region = None
        mock_agent.model = MagicMock(spec=[])  # No config attr
        mock_boto3.Session.return_value.region_name = "ap-southeast-1"

        result = target._resolve_region(mock_agent, config)
        assert result == "ap-southeast-1"


class TestAgentCoreDestroy:
    @patch("strands.deploy._agentcore.boto3")
    def test_destroy_deletes_runtime_and_role(self, mock_boto3, target, state_manager):
        state_manager.save(
            "my-agent",
            DeployState(
                target="agentcore",
                region="us-west-2",
                agent_runtime_id="rt-123",
                role_arn="arn:aws:iam::123:role/strands-my-agent-agentcore-role",
            ),
        )

        mock_control = MagicMock()
        mock_iam = MagicMock()
        mock_iam.list_role_policies.return_value = {"PolicyNames": ["strands-agentcore-execution"]}

        def client_factory(service, **kwargs):
            return {"bedrock-agentcore-control": mock_control, "iam": mock_iam}.get(service, MagicMock())

        mock_boto3.client.side_effect = client_factory

        target.destroy("my-agent", state_manager)

        mock_control.delete_agent_runtime.assert_called_once_with(agentRuntimeId="rt-123")
        mock_iam.delete_role.assert_called_once()
        assert state_manager.load("my-agent") is None

    @patch("strands.deploy._agentcore.boto3")
    def test_destroy_nonexistent_is_noop(self, mock_boto3, target, state_manager):
        target.destroy("nonexistent", state_manager)
        mock_boto3.client.assert_not_called()
