"""Abstract base class for deployment targets."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from . import DeployConfig, DeployResult
    from ._state import StateManager


class DeployTarget(ABC):
    """Strategy interface for deployment targets.

    Each target (AgentCore, Lambda, etc.) implements this interface
    to handle the specifics of packaging, provisioning, and updating
    cloud resources for a Strands agent.
    """

    @abstractmethod
    def validate(self, config: "DeployConfig") -> None:
        """Validate that the deployment can proceed.

        Check prerequisites like installed packages, AWS credentials, etc.
        Raises DeployException if validation fails.
        """
        ...

    @abstractmethod
    def deploy(self, agent: "Agent", config: "DeployConfig", state_manager: "StateManager") -> "DeployResult":
        """Deploy the agent to this target.

        Handles the full lifecycle: packaging, IAM, provisioning, and state tracking.
        """
        ...

    @abstractmethod
    def destroy(self, name: str, state_manager: "StateManager", region: str | None = None) -> None:
        """Tear down deployed resources for the named deployment."""
        ...
