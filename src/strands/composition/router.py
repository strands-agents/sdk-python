"""Event Router — Derives event flow graph from manifest declarations.

Computes how events flow between agents based on their declared
output_contract.events_produced and input_contract.events_consumed.
"""

import logging
from dataclasses import dataclass

from strands.composition.registry import CompositionRegistry

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """A resolved event route: source agent produces event, target agents consume it.

    Attributes:
        event: The event type being routed.
        source: Agent name that produces this event.
        targets: Agent names that consume this event.
    """

    event: str
    source: str
    targets: list[str]

    def __eq__(self, other: object) -> bool:
        """Equality based on event, source, and targets (order-independent)."""
        if not isinstance(other, Route):
            return False
        return (
            self.event == other.event
            and self.source == other.source
            and set(self.targets) == set(other.targets)
        )

    def __hash__(self) -> int:
        """Hash based on event, source, and targets."""
        return hash((self.event, self.source, frozenset(self.targets)))

    def __repr__(self) -> str:
        """Readable representation."""
        targets_str = ", ".join(self.targets)
        return f"Route({self.source} --[{self.event}]--> [{targets_str}])"


class EventRouter:
    """Computes event routing from manifest declarations.

    Given a registry of agents, determines how events flow between them
    based on output_contract.events_produced and input_contract.events_consumed.
    No manual wiring needed — routes emerge from contracts.

    Example:
        >>> router = EventRouter(registry)
        >>> routes = router.resolve_routes()
        >>> print(router.visualize())
        Event Flow:
          rca_agent --[FaultEvent]--> impact_agent
    """

    def __init__(self, registry: CompositionRegistry) -> None:
        """Initialize with a composition registry.

        Args:
            registry: The registry containing agents and their manifests.
        """
        self._registry = registry

    def resolve_routes(self) -> list[Route]:
        """Compute all event routes across registered agents.

        For each event type produced by any agent, find all agents that
        consume it and create a route.

        Returns:
            List of Route objects representing the event flow graph.
        """
        routes: list[Route] = []

        produced: dict[str, list[str]] = {}
        for manifest in self._registry.manifests:
            for event in manifest.output_contract.events_produced:
                produced.setdefault(event, []).append(manifest.name)

        for event_type, producers in produced.items():
            consumers = self._registry.consumers_of(event_type)
            if consumers:
                for producer in producers:
                    targets = [c for c in consumers if c != producer]
                    if targets:
                        routes.append(Route(event=event_type, source=producer, targets=targets))

        return routes

    def routes_from(self, agent_name: str) -> list[Route]:
        """Get all routes originating from a specific agent.

        Args:
            agent_name: The agent name to filter by.

        Returns:
            List of routes where this agent is the source.
        """
        return [r for r in self.resolve_routes() if r.source == agent_name]

    def routes_to(self, agent_name: str) -> list[Route]:
        """Get all routes targeting a specific agent.

        Args:
            agent_name: The agent name to filter by.

        Returns:
            List of routes where this agent is a target.
        """
        return [r for r in self.resolve_routes() if agent_name in r.targets]

    def dependency_order(self) -> list[list[str]]:
        """Topological sort of agents by event dependencies.

        Returns layers where agents in layer 0 have no event dependencies,
        agents in layer 1 depend only on layer 0 outputs, etc.

        Returns:
            List of layers, each layer is a list of agent names that can
            execute in parallel.
        """
        manifests = {m.name: m for m in self._registry.manifests}

        deps: dict[str, set[str]] = {name: set() for name in manifests}
        for manifest in manifests.values():
            for event in manifest.input_contract.events_consumed:
                producers = self._registry.producers_of(event)
                for p in producers:
                    if p != manifest.name:
                        deps[manifest.name].add(p)

        layers: list[list[str]] = []
        remaining = {k: set(v) for k, v in deps.items()}

        while remaining:
            layer = [n for n, d in remaining.items() if not d]
            if not layer:
                # Cycle detected — include remaining as final layer
                layer = list(remaining.keys())
                layers.append(layer)
                break
            layers.append(layer)
            for n in layer:
                del remaining[n]
            for d in remaining.values():
                d -= set(layer)

        return layers

    def visualize(self) -> str:
        """ASCII visualization of the event flow graph.

        Returns:
            Formatted string showing all event routes.
        """
        routes = self.resolve_routes()
        if not routes:
            return "(no routes)"

        lines: list[str] = []
        for route in routes:
            targets_str = ", ".join(route.targets)
            lines.append(f"  {route.source} --[{route.event}]--> {targets_str}")

        return "Event Flow:\n" + "\n".join(sorted(lines))
