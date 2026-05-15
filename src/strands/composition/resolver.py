"""Binding Resolver — Evaluates agent manifests against available resources.

Determines whether each agent's input contract can be satisfied given the
currently available events, features, tools, and knowledge bases.
"""

import logging
from dataclasses import dataclass, field

from strands.agent.manifest import AgentManifest

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """A single unresolved dependency.

    Attributes:
        type: The resource type ("event", "feature", "tool", "knowledge_base").
        name: The name of the missing resource.
        description: Optional human-readable description.
    """

    type: str
    name: str
    description: str = ""


@dataclass
class BindingStatus:
    """Result of evaluating a manifest against available resources.

    Attributes:
        agent_name: Name of the evaluated agent.
        state: "active" if all dependencies met, "dormant" if gaps exist.
        resolved_events: Events that are available.
        resolved_features: Features that are available.
        resolved_tools: Tools that are available.
        resolved_knowledge_bases: Knowledge bases that are available.
        gaps: List of unresolved dependencies.
    """

    agent_name: str
    state: str
    resolved_events: list[str] = field(default_factory=list)
    resolved_features: list[str] = field(default_factory=list)
    resolved_tools: list[str] = field(default_factory=list)
    resolved_knowledge_bases: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    def gap_report(self) -> str:
        """Human-readable gap report.

        Returns:
            Formatted string showing agent state and any unresolved dependencies.
        """
        if not self.gaps:
            return f"{self.agent_name}: all bindings resolved (active)"
        lines = [f"{self.agent_name}: {len(self.gaps)} unresolved binding(s) (dormant)"]
        for gap in self.gaps:
            lines.append(f"  - {gap.type}: {gap.name}")
        return "\n".join(lines)


class BindingResolver:
    """Evaluates agent manifests against available resources.

    Given a set of available events, features, tools, and knowledge bases,
    determines whether each agent's input contract can be satisfied.

    Example:
        >>> resolver = BindingResolver(
        ...     available_events=["AlarmEvent"],
        ...     available_features=["device_health"],
        ...     available_tools=["query_tool"],
        ... )
        >>> status = resolver.evaluate(agent.manifest)
        >>> print(status.state)  # "active" or "dormant"
    """

    def __init__(
        self,
        available_events: list[str] | None = None,
        available_features: list[str] | None = None,
        available_tools: list[str] | None = None,
        available_knowledge_bases: list[str] | None = None,
    ) -> None:
        """Initialize with available resources.

        Args:
            available_events: Event types currently available.
            available_features: Feature names currently available.
            available_tools: Tool capability names currently registered.
            available_knowledge_bases: Knowledge base names currently available.
        """
        self._events: set[str] = set(available_events or [])
        self._features: set[str] = set(available_features or [])
        self._tools: set[str] = set(available_tools or [])
        self._knowledge_bases: set[str] = set(available_knowledge_bases or [])

    @property
    def available_events(self) -> set[str]:
        """Currently available events."""
        return self._events.copy()

    @property
    def available_features(self) -> set[str]:
        """Currently available features."""
        return self._features.copy()

    @property
    def available_tools(self) -> set[str]:
        """Currently available tools."""
        return self._tools.copy()

    @property
    def available_knowledge_bases(self) -> set[str]:
        """Currently available knowledge bases."""
        return self._knowledge_bases.copy()

    def add_event(self, event: str) -> None:
        """Register a new available event."""
        self._events.add(event)

    def remove_event(self, event: str) -> None:
        """Remove an available event."""
        self._events.discard(event)

    def add_feature(self, feature: str) -> None:
        """Register a new available feature."""
        self._features.add(feature)

    def remove_feature(self, feature: str) -> None:
        """Remove an available feature."""
        self._features.discard(feature)

    def add_tool(self, tool: str) -> None:
        """Register a new available tool."""
        self._tools.add(tool)

    def remove_tool(self, tool: str) -> None:
        """Remove an available tool."""
        self._tools.discard(tool)

    def add_knowledge_base(self, kb: str) -> None:
        """Register a new available knowledge base."""
        self._knowledge_bases.add(kb)

    def remove_knowledge_base(self, kb: str) -> None:
        """Remove an available knowledge base."""
        self._knowledge_bases.discard(kb)

    def evaluate(self, manifest: AgentManifest) -> BindingStatus:
        """Evaluate a manifest against current available resources.

        Args:
            manifest: The agent manifest to evaluate.

        Returns:
            BindingStatus indicating whether the agent can activate
            (all dependencies met) or is dormant (gaps exist).
        """
        gaps: list[Gap] = []
        resolved_events: list[str] = []
        resolved_features: list[str] = []
        resolved_tools: list[str] = []
        resolved_kbs: list[str] = []

        for event in manifest.input_contract.events_consumed:
            if event in self._events:
                resolved_events.append(event)
            else:
                gaps.append(Gap(type="event", name=event))

        for feature in manifest.input_contract.features_required:
            if feature in self._features:
                resolved_features.append(feature)
            else:
                gaps.append(Gap(type="feature", name=feature))

        for tool in manifest.input_contract.tool_capabilities:
            if tool in self._tools:
                resolved_tools.append(tool)
            else:
                gaps.append(Gap(type="tool", name=tool))

        for kb in manifest.input_contract.knowledge_bases:
            if kb in self._knowledge_bases:
                resolved_kbs.append(kb)
            else:
                gaps.append(Gap(type="knowledge_base", name=kb))

        state = "active" if not gaps else "dormant"

        return BindingStatus(
            agent_name=manifest.name,
            state=state,
            resolved_events=resolved_events,
            resolved_features=resolved_features,
            resolved_tools=resolved_tools,
            resolved_knowledge_bases=resolved_kbs,
            gaps=gaps,
        )

    def evaluate_all(self, manifests: list[AgentManifest]) -> list[BindingStatus]:
        """Evaluate multiple manifests.

        Args:
            manifests: List of manifests to evaluate.

        Returns:
            List of BindingStatus, one per manifest.
        """
        return [self.evaluate(m) for m in manifests]

    def what_would_activate(
        self, resource_type: str, resource_name: str, manifests: list[AgentManifest]
    ) -> list[str]:
        """Preview which dormant agents would activate if a resource were added.

        Does not mutate state.

        Args:
            resource_type: One of "event", "feature", "tool", "knowledge_base".
            resource_name: Name of the resource to simulate adding.
            manifests: List of manifests to check.

        Returns:
            List of agent names that would transition from dormant to active.
        """
        attr_name = f"_{resource_type}s"
        original = getattr(self, attr_name).copy()

        getattr(self, attr_name).add(resource_name)

        would_activate = []
        for manifest in manifests:
            status = self.evaluate(manifest)
            if status.state == "active":
                # Check if it was dormant before
                getattr(self, attr_name).discard(resource_name)
                prev_status = self.evaluate(manifest)
                getattr(self, attr_name).add(resource_name)
                if prev_status.state == "dormant":
                    would_activate.append(manifest.name)

        setattr(self, attr_name, original)
        return would_activate
