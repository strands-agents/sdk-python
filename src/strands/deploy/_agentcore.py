"""AgentCore deployment target using the bedrock-agentcore-starter-toolkit."""

import logging
import os
import tempfile
from typing import TYPE_CHECKING

from ._base import DeployTarget
from ._constants import get_python_runtime
from ._exceptions import DeployException, DeployTargetException
from ._packaging import generate_agentcore_entrypoint
from ._state import DeployState

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from . import DeployConfig, DeployResult
    from ._state import StateManager

logger = logging.getLogger(__name__)


def _get_runtime_class():
    """Import and return the Runtime class from the starter toolkit.

    Raises DeployException if the package is not installed.
    """
    try:
        from bedrock_agentcore_starter_toolkit import Runtime
    except ImportError as e:
        raise DeployException(
            "bedrock-agentcore-starter-toolkit is required for AgentCore deployment. "
            "Install it with: pip install 'strands-agents[deploy]'"
        ) from e
    return Runtime


class AgentCoreTarget(DeployTarget):
    """Deploy a Strands agent to AWS Bedrock AgentCore.

    Delegates to the bedrock-agentcore-starter-toolkit Runtime class
    for IAM, S3, runtime creation, polling, and endpoint management.
    """

    def validate(self, config: "DeployConfig") -> None:
        """Check that the starter toolkit is installed and AWS credentials are available."""
        _get_runtime_class()

        try:
            import boto3

            sts = boto3.client("sts")
            sts.get_caller_identity()
        except Exception as e:
            raise DeployException(f"AWS credentials not configured: {e}") from e

    def deploy(self, agent: "Agent", config: "DeployConfig", state_manager: "StateManager") -> "DeployResult":
        """Deploy agent to AgentCore via the starter toolkit."""
        from . import DeployResult

        Runtime = _get_runtime_class()
        region = self._resolve_region(agent, config)
        print(f"  Region: {region}")

        # Generate the entrypoint file into a temp directory
        entrypoint_code = generate_agentcore_entrypoint(agent)
        work_dir = tempfile.mkdtemp(prefix="strands-deploy-")
        entrypoint_path = os.path.join(work_dir, "_strands_entrypoint.py")
        with open(entrypoint_path, "w") as f:
            f.write(entrypoint_code)

        existing = state_manager.load(config.name)
        is_update = bool(existing and existing.get("agent_runtime_id"))

        runtime = Runtime()
        configure_kwargs: dict = {
            "entrypoint": entrypoint_path,
            "agent_name": f"strands-{config.name}",
            "deployment_type": "direct_code_deploy",
            "runtime_type": get_python_runtime(),
            "region": region,
            "non_interactive": True,
        }

        if config.environment_variables:
            configure_kwargs["environment_variables"] = config.environment_variables
        if config.description:
            configure_kwargs["description"] = config.description

        try:
            runtime.configure(**configure_kwargs)

            if is_update:
                print(f"  Updating AgentCore Runtime: strands-{config.name}")
            else:
                print(f"  Creating AgentCore Runtime: strands-{config.name}")

            launch_result = runtime.launch()
        except Exception as e:
            raise DeployTargetException("agentcore", str(e), cause=e) from e

        # Extract result fields from the toolkit's launch result
        runtime_id = getattr(launch_result, "agent_runtime_id", None) or ""
        runtime_arn = getattr(launch_result, "agent_runtime_arn", None) or ""
        endpoint_arn = getattr(launch_result, "agent_runtime_endpoint_arn", None) or ""
        role_arn = getattr(launch_result, "role_arn", None) or ""

        # Save state
        state = DeployState(
            target="agentcore",
            region=region,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
            agent_runtime_endpoint_arn=endpoint_arn,
            role_arn=role_arn,
        )
        state_manager.save(config.name, state)

        return DeployResult(
            target="agentcore",
            name=config.name,
            region=region,
            created=not is_update,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
            agent_runtime_endpoint_arn=endpoint_arn,
            role_arn=role_arn,
        )

    def destroy(self, name: str, state_manager: "StateManager", region: str | None = None) -> None:
        """Tear down AgentCore resources via the starter toolkit."""
        existing = state_manager.load(name)
        if not existing:
            print(f"  No deployment found for '{name}'")
            return

        Runtime = _get_runtime_class()
        deploy_region = region or existing.get("region", "us-east-1")

        try:
            runtime = Runtime()
            runtime.configure(
                agent_name=f"strands-{name}",
                region=deploy_region,
                non_interactive=True,
            )
            print(f"  Destroying AgentCore Runtime: strands-{name}")
            runtime.destroy()
        except Exception as e:
            logger.warning("Failed to destroy runtime for '%s': %s", name, e)

        state_manager.delete(name)
        print(f"  Deployment '{name}' destroyed")

    def _resolve_region(self, agent: "Agent", config: "DeployConfig") -> str:
        """Resolve the AWS region from config, agent model, or environment."""
        if config.region:
            return config.region

        model = agent.model
        if hasattr(model, "config") and hasattr(model.config, "get"):
            model_region: str | None = model.config.get("region")
            if model_region:
                return model_region

        import boto3

        session = boto3.Session()
        return session.region_name or os.environ.get("AWS_REGION") or "us-east-1"
