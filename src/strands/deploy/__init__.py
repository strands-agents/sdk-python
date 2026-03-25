"""Agent deployment module for pushing Strands agents to AWS.

Usage:
    agent = Agent(model=BedrockModel(), tools=[my_tool])
    result = agent.deploy(target="agentcore", name="my-agent")
    print(result.agent_runtime_arn)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ._exceptions import DeployException, DeployPackagingException, DeployStateException, DeployTargetException

if TYPE_CHECKING:
    from ..agent.agent import Agent

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
    auth: Literal["public", "iam"] = "public"
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
    agent_runtime_endpoint_arn: str | None = None
    role_arn: str | None = None


def deploy(agent: "Agent", config: DeployConfig) -> DeployResult:
    """Deploy an agent to the specified target.

    Args:
        agent: The Strands Agent instance to deploy.
        config: Deployment configuration.

    Returns:
        DeployResult with ARNs and endpoint information.

    Raises:
        DeployException: If deployment fails.
    """
    from ._agentcore import AgentCoreTarget
    from ._state import StateManager

    targets = {
        "agentcore": AgentCoreTarget,
    }

    target_cls = targets.get(config.target)
    if target_cls is None:
        raise DeployException(
            f"Unknown deploy target: '{config.target}'. Supported targets: {', '.join(targets.keys())}"
        )

    target = target_cls()
    target.validate(config)

    state_manager = StateManager()

    print(f"\nDeploying agent '{config.name}' to {config.target}...")
    result = target.deploy(agent, config, state_manager)

    print()
    if result.agent_runtime_arn:
        print(f"  Runtime ARN: {result.agent_runtime_arn}")
    if result.agent_runtime_endpoint_arn:
        print(f"  Endpoint ARN: {result.agent_runtime_endpoint_arn}")
    print(f"  Status: {'Created' if result.created else 'Updated'}")
    print("\n  Deploy state saved to .strands/state.json")

    return result
