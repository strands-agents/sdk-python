"""Lifecycle Manager — Manages dormant/active transitions reactively.

When resources are added or removed, re-evaluates all agents and
transitions them between dormant and active states.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from strands.composition.registry import CompositionRegistry
from strands.composition.resolver import BindingResolver, BindingStatus

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Record of an agent state change.

    Attributes:
        agent_name: Name of the agent that transitioned.
        from_state: Previous state ("active" or "dormant").
        to_state: New state ("active" or "dormant").
        timestamp: When the transition occurred.
        reason: What triggered the transition.
    """

    agent_name: str
    from_state: str
    to_state: str
    timestamp: datetime
    reason: str


@dataclass
class AgentState:
    """Current state of an agent in the composition.

    Attributes:
        name: Agent name.
        state: Current state ("active" or "dormant").
        binding_status: Full binding evaluation result.
    """

    name: str
    state: str
    binding_status: BindingStatus


class LifecycleManager:
    """Manages agent lifecycle based on resource availability.

    When resources are added or removed, re-evaluates all agents and
    transitions them between dormant and active states. Maintains a
    history of all transitions for audit.

    Example:
        >>> manager = LifecycleManager(registry, resolver)
        >>> print(manager.active_agents)
        ['rca_agent']
        >>> transitions = manager.add_resource("feature", "service_mapping")
        >>> print(transitions[0].agent_name, transitions[0].to_state)
        impact_agent active
    """

    def __init__(self, registry: CompositionRegistry, resolver: BindingResolver) -> None:
        """Initialize with a registry and resolver.

        Performs initial evaluation of all registered agents.

        Args:
            registry: The composition registry containing agents.
            resolver: The binding resolver with current available resources.
        """
        self._registry = registry
        self._resolver = resolver
        self._states: dict[str, str] = {}
        self._history: list[StateTransition] = []

        self._evaluate_all()

    def _evaluate_all(self) -> list[StateTransition]:
        """Re-evaluate all agents and return any state transitions."""
        transitions: list[StateTransition] = []
        for manifest in self._registry.manifests:
            status = self._resolver.evaluate(manifest)
            old_state = self._states.get(manifest.name, "dormant")
            new_state = status.state

            if old_state != new_state:
                transition = StateTransition(
                    agent_name=manifest.name,
                    from_state=old_state,
                    to_state=new_state,
                    timestamp=datetime.now(),
                    reason="re-evaluation",
                )
                transitions.append(transition)
                self._history.append(transition)
                logger.info(
                    "agent=%s | %s -> %s (%s)",
                    manifest.name,
                    old_state,
                    new_state,
                    transition.reason,
                )

            self._states[manifest.name] = new_state

        return transitions

    def add_resource(self, resource_type: str, name: str) -> list[StateTransition]:
        """Add a resource and return any resulting state transitions.

        Args:
            resource_type: One of "event", "feature", "tool", "knowledge_base".
            name: Name of the resource being added.

        Returns:
            List of state transitions triggered by this addition.
        """
        add_method = getattr(self._resolver, f"add_{resource_type}")
        add_method(name)

        transitions = self._evaluate_all()
        for t in transitions:
            t.reason = f"{resource_type} '{name}' added"

        return transitions

    def remove_resource(self, resource_type: str, name: str) -> list[StateTransition]:
        """Remove a resource and return any resulting state transitions.

        Args:
            resource_type: One of "event", "feature", "tool", "knowledge_base".
            name: Name of the resource being removed.

        Returns:
            List of state transitions triggered by this removal.
        """
        remove_method = getattr(self._resolver, f"remove_{resource_type}")
        remove_method(name)

        transitions = self._evaluate_all()
        for t in transitions:
            t.reason = f"{resource_type} '{name}' removed"

        return transitions

    def get_state(self, agent_name: str) -> AgentState:
        """Get current state of an agent.

        Args:
            agent_name: The agent name to query.

        Returns:
            AgentState with current state and full binding status.

        Raises:
            KeyError: If the agent is not registered.
        """
        manifest = self._registry.get_manifest(agent_name)
        if manifest is None:
            raise KeyError(f"Agent '{agent_name}' not found in registry")
        status = self._resolver.evaluate(manifest)
        return AgentState(name=agent_name, state=status.state, binding_status=status)

    @property
    def active_agents(self) -> list[str]:
        """List of agent names currently in active state."""
        return [name for name, state in self._states.items() if state == "active"]

    @property
    def dormant_agents(self) -> list[str]:
        """List of agent names currently in dormant state."""
        return [name for name, state in self._states.items() if state == "dormant"]

    @property
    def history(self) -> list[StateTransition]:
        """Full history of state transitions."""
        return list(self._history)

    def summary(self) -> str:
        """Human-readable summary of current composition state.

        Returns:
            Formatted string showing active/dormant agents and their gaps.
        """
        lines = [f"Active ({len(self.active_agents)}): {', '.join(self.active_agents) or '(none)'}"]
        lines.append(f"Dormant ({len(self.dormant_agents)}):")
        for name in self.dormant_agents:
            state = self.get_state(name)
            gaps = [f"{g.type}:{g.name}" for g in state.binding_status.gaps]
            lines.append(f"  {name}: needs [{', '.join(gaps)}]")
        return "\n".join(lines)
