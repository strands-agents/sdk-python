"""Composition Registry — Local catalog of agents participating in a composition.

Provides lookup and query capabilities across registered agents and their manifests.
"""

import logging
from typing import Optional

from strands.agent.manifest import AgentManifest

logger = logging.getLogger(__name__)


class CompositionRegistry:
    """Local registry of agents participating in a composition.

    Holds agent references and their manifests, enabling dependency resolution
    and route computation across the set of registered agents.

    Example:
        >>> from strands.composition import CompositionRegistry
        >>> registry = CompositionRegistry()
        >>> registry.register(my_agent)
        >>> registry.producers_of("FaultEvent")
        ['rca_agent']
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._agents: dict[str, object] = {}
        self._manifests: dict[str, AgentManifest] = {}

    def register(self, agent: object) -> None:
        """Register an agent. Requires agent to have a manifest attribute.

        Args:
            agent: An Agent instance with a manifest attribute.

        Raises:
            ValueError: If the agent has no manifest.
        """
        manifest = getattr(agent, "manifest", None)
        if manifest is None:
            raise ValueError("Cannot register agent without a manifest")
        if not isinstance(manifest, AgentManifest):
            raise ValueError("Agent manifest must be an AgentManifest instance")

        self._agents[manifest.name] = agent
        self._manifests[manifest.name] = manifest
        logger.debug("agent=%s | registered in composition registry", manifest.name)

    def register_manifest(self, manifest: AgentManifest) -> None:
        """Register a manifest without an agent instance.

        Useful for validating compositions before agents are instantiated.

        Args:
            manifest: An AgentManifest instance.
        """
        self._manifests[manifest.name] = manifest
        logger.debug("manifest=%s | registered in composition registry (no agent instance)", manifest.name)

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry.

        Args:
            name: The manifest name of the agent to remove.

        Raises:
            KeyError: If the agent is not registered.
        """
        if name not in self._manifests:
            raise KeyError(f"Agent '{name}' not found in registry")
        self._manifests.pop(name)
        self._agents.pop(name, None)
        logger.debug("agent=%s | unregistered from composition registry", name)

    def get(self, name: str) -> Optional[object]:
        """Get an agent by manifest name.

        Args:
            name: The manifest name.

        Returns:
            The agent instance, or None if not found.
        """
        return self._agents.get(name)

    def get_manifest(self, name: str) -> Optional[AgentManifest]:
        """Get a manifest by name.

        Args:
            name: The manifest name.

        Returns:
            The AgentManifest instance, or None if not found.
        """
        return self._manifests.get(name)

    @property
    def agents(self) -> list[object]:
        """All registered agent instances."""
        return list(self._agents.values())

    @property
    def manifests(self) -> list[AgentManifest]:
        """All registered manifests."""
        return list(self._manifests.values())

    @property
    def names(self) -> list[str]:
        """All registered agent names."""
        return list(self._manifests.keys())

    def __len__(self) -> int:
        """Number of registered manifests."""
        return len(self._manifests)

    def __contains__(self, name: str) -> bool:
        """Check if an agent name is registered."""
        return name in self._manifests

    def producers_of(self, event_type: str) -> list[str]:
        """Find agents that produce a given event type.

        Args:
            event_type: The event type to search for.

        Returns:
            List of agent names that produce this event type.
        """
        return [
            m.name
            for m in self._manifests.values()
            if event_type in m.output_contract.events_produced
        ]

    def consumers_of(self, event_type: str) -> list[str]:
        """Find agents that consume a given event type.

        Args:
            event_type: The event type to search for.

        Returns:
            List of agent names that consume this event type.
        """
        return [
            m.name
            for m in self._manifests.values()
            if event_type in m.input_contract.events_consumed
        ]

    def providers_of_feature(self, feature_name: str) -> list[str]:
        """Find agents that produce a given feature.

        Args:
            feature_name: The feature name to search for.

        Returns:
            List of agent names that produce this feature.
        """
        return [
            m.name
            for m in self._manifests.values()
            if feature_name in m.output_contract.features_produced
        ]
