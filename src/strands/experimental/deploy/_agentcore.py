"""AgentCore deployment target using the bedrock-agentcore-starter-toolkit."""

import logging
import os
from typing import TYPE_CHECKING

from ._base import DeployTarget
from ._constants import (
    AGENTCORE_BASE_REQUIREMENTS,
    DEPLOYMENT_TYPE,
    ENTRYPOINT_FILENAME,
    TOOLKIT_BUILD_ARTIFACTS,
    agentcore_runtime_name,
    get_python_runtime,
)
from ._exceptions import DeployException, DeployTargetException
from ._packaging import generate_agentcore_entrypoint
from ._state import DeployState

if TYPE_CHECKING:
    from ...agent.agent import Agent
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


def _inject_user_agent():
    """Best-effort injection of strands-agents-deploy into the toolkit's user agent string."""
    try:
        from bedrock_agentcore_starter_toolkit.services import runtime as _toolkit_runtime

        _original_fn = _toolkit_runtime._get_user_agent
        if "strands-agents-deploy" not in _original_fn():
            _toolkit_runtime._get_user_agent = lambda: _original_fn() + " strands-agents-deploy"
    except Exception:
        pass


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
        runtime_name = agentcore_runtime_name(config.name)
        logger.info("region=<%s> | resolved deployment region", region)

        # Generate the entrypoint file in CWD so the toolkit includes it in the deployment zip
        entrypoint_code = generate_agentcore_entrypoint(agent)
        entrypoint_path = os.path.join(os.getcwd(), ENTRYPOINT_FILENAME)
        with open(entrypoint_path, "w") as f:
            f.write(entrypoint_code)

        # Build requirements list: our deps + any existing project requirements
        requirements = list(AGENTCORE_BASE_REQUIREMENTS)
        existing_reqs_path = os.path.join(os.getcwd(), "requirements.txt")
        had_existing_reqs = os.path.exists(existing_reqs_path)
        if had_existing_reqs:
            with open(existing_reqs_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        requirements.append(line)

        existing = state_manager.load(config.name)
        is_update = bool(existing and existing.get("agent_runtime_id"))

        runtime = Runtime()
        configure_kwargs: dict = {
            "entrypoint": entrypoint_path,
            "agent_name": runtime_name,
            "deployment_type": DEPLOYMENT_TYPE,
            "runtime_type": get_python_runtime(),
            "region": region,
            "non_interactive": True,
            "auto_create_execution_role": True,
            "requirements": requirements,
        }

        if config.environment_variables:
            configure_kwargs["environment_variables"] = config.environment_variables
        if config.description:
            configure_kwargs["description"] = config.description

        try:
            runtime.configure(**configure_kwargs)

            if is_update:
                logger.info("runtime_name=<%s> | updating agentcore runtime", runtime_name)
            else:
                logger.info("runtime_name=<%s> | creating agentcore runtime", runtime_name)

            _inject_user_agent()
            launch_result = runtime.launch()
        except Exception as e:
            raise DeployTargetException("agentcore", str(e), cause=e) from e
        finally:
            cwd = os.path.dirname(entrypoint_path)
            cleanup = [entrypoint_path]
            if not had_existing_reqs:
                cleanup.append(os.path.join(cwd, "requirements.txt"))
            cleanup.extend(os.path.join(cwd, artifact) for artifact in TOOLKIT_BUILD_ARTIFACTS)
            for path in cleanup:
                try:
                    os.remove(path)
                except OSError:
                    pass

        # Extract result fields from the toolkit's LaunchResult
        runtime_id = launch_result.agent_id or ""
        runtime_arn = launch_result.agent_arn or ""

        # Save state
        state = DeployState(
            target="agentcore",
            region=region,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
        )
        state_manager.save(config.name, state)

        return DeployResult(
            target="agentcore",
            name=config.name,
            region=region,
            created=not is_update,
            agent_runtime_id=runtime_id,
            agent_runtime_arn=runtime_arn,
        )

    def destroy(self, name: str, state_manager: "StateManager", region: str | None = None) -> None:
        """Tear down AgentCore resources via the starter toolkit."""
        existing = state_manager.load(name)
        if not existing:
            logger.info("name=<%s> | no deployment found, skipping destroy", name)
            return

        Runtime = _get_runtime_class()
        deploy_region = region or existing.get("region", "us-east-1")
        runtime_name = agentcore_runtime_name(name)

        try:
            runtime = Runtime()
            runtime.configure(
                agent_name=runtime_name,
                region=deploy_region,
                non_interactive=True,
            )
            _inject_user_agent()
            logger.info("runtime_name=<%s> | destroying agentcore runtime", runtime_name)
            runtime.destroy()
        except Exception as e:
            logger.warning("name=<%s> | failed to destroy runtime: %s", name, e)

        state_manager.delete(name)
        logger.info("name=<%s> | deployment destroyed", name)

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
