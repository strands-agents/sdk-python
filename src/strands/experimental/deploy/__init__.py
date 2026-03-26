"""Experimental agent deployment module for pushing Strands agents to AWS.

Usage:
    from strands.experimental import deploy

    agent = Agent(model=BedrockModel(), tools=[my_tool])
    result = deploy(agent, target="agentcore", name="my-agent")
    print(result.agent_runtime_arn)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ._exceptions import DeployException, DeployPackagingException, DeployStateException, DeployTargetException

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)

__all__ = [
    "DeployConfig",
    "DeployException",
    "DeployPackagingException",
    "DeployResult",
    "DeployStateException",
    "DeployTargetException",
    "deploy",
]


@dataclass
class DeployConfig:
    """Configuration for an agent deployment."""

    target: Literal["agentcore"]
    name: str
    region: str | None = None
    description: str | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)


@dataclass
class DeployResult:
    """Result of a deployment operation."""

    target: str
    name: str
    region: str
    created: bool = True
    # AgentCore fields
    agent_runtime_id: str | None = None
    agent_runtime_arn: str | None = None


def deploy(
    agent: "Agent",
    target: Literal["agentcore"] = "agentcore",
    *,
    name: str | None = None,
    region: str | None = None,
    description: str | None = None,
    environment_variables: dict[str, str] | None = None,
) -> DeployResult:
    """Deploy an agent to a cloud target.

    Packages the agent's code and configuration, provisions cloud infrastructure,
    and returns endpoint information. State is tracked in .strands_deploy/state.json
    so subsequent calls update the existing deployment.

    Args:
        agent: The Strands Agent instance to deploy.
        target: Deployment target. Currently only "agentcore" (Bedrock AgentCore) is supported.
        name: Name for the deployed resource. Defaults to a sanitized version of agent.name.
        region: AWS region. Auto-detected from the model config, boto3 session, or AWS_REGION env var.
        description: Description for the deployed runtime.
        environment_variables: Environment variables to set in the runtime.

    Returns:
        DeployResult with ARNs and endpoint information.

    Raises:
        DeployException: If deployment fails.

    Note:
        **Packaging:** The deployment packages all files in the current working directory.
        The caller's source file is copied into the entrypoint with the ``deploy()`` call
        stripped out, so all tools, plugins, hooks, and Agent parameters are preserved.

        **Imports:** Absolute imports from the CWD root and relative imports within the
        caller's directory are supported. For projects that rely on custom ``PYTHONPATH``
        or editable installs, run ``deploy()`` from the project root.

        **Dependencies:** ``bedrock-agentcore`` and ``strands-agents`` are automatically
        included. For additional dependencies, provide a ``requirements.txt`` file in the
        current working directory. Dependencies in ``pyproject.toml`` are not yet
        automatically included.

    Example::

        from strands.experimental import deploy

        agent = Agent(model="us.anthropic.claude-sonnet-4-20250514")
        result = deploy(agent, target="agentcore", name="my-agent")
        print(result.agent_runtime_arn)
    """
    from ._agentcore import AgentCoreTarget
    from ._state import StateManager

    targets = {
        "agentcore": AgentCoreTarget,
    }

    target_cls = targets.get(target)
    if target_cls is None:
        raise DeployException(
            f"Unknown deploy target: '{target}'. Supported targets: {', '.join(targets.keys())}"
        )

    deploy_name = name or re.sub(r"[^a-zA-Z0-9_]", "_", agent.name.lower()).strip("_")[:40]
    config = DeployConfig(
        target=target,
        name=deploy_name,
        region=region,
        description=description,
        environment_variables=environment_variables or {},
    )

    target_impl = target_cls()
    target_impl.validate(config)

    state_manager = StateManager()

    print(f"\nDeploying agent '{config.name}' to {config.target}...")
    result = target_impl.deploy(agent, config, state_manager)

    print()
    if result.agent_runtime_arn:
        print(f"  Runtime ARN: {result.agent_runtime_arn}")
    print(f"  Status: {'Created' if result.created else 'Updated'}")
    print("\n  Deploy state saved to .strands_deploy/state.json")

    return result
