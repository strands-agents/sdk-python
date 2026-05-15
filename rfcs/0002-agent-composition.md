# RFC 0002: Agent Composition — Declarative Multi-Agent Coordination

## Summary

Add a `strands.composition` module to the SDK that resolves agent dependencies and computes coordination from manifest declarations. Given a set of agents with manifests (RFC 0001) and a set of available resources, the module determines which agents can activate, which are blocked, what's missing, and how they connect — without requiring an orchestrator, workflow definition, or custom wiring code.

The approach is **declarative**: agents declare what they need (events, features, tools, knowledge bases) and what they produce. The composition engine resolves the full dependency graph — not just event routing, but data readiness, tool availability, and knowledge base access. The coordination structure emerges from declarations, not from imperative wiring.

## Motivation

### Problem

RFC 0001 introduced `AgentManifest` — agents can now declare what they need and what they produce. But declarations alone don't compose agents. Today, after defining manifests, a developer must still:

1. **Manually check dependencies**: "Can this agent run? Are its required features available? Are the tools it needs bound? Are the events it consumes being produced by anyone?"
2. **Manually wire coordination**: "Agent A produces EventX, Agent B consumes EventX — I need to write the glue that connects them."
3. **Manually manage lifecycle**: "Agent C can't run yet because its data source isn't ready. I'll deploy it later and remember to connect it."

This manual work is exactly what declarative manifests were designed to eliminate. The missing piece is the **resolution engine** — the logic that reads manifests, evaluates them against available resources, and produces the composition graph automatically.

### The Core Insight

In multi-agent orchestration, someone writes the DAG — a workflow definition, a Step Function, a coordinator that calls agents in sequence. That's imperative coordination: you specify HOW agents connect.

In declarative composition, you specify WHAT each agent needs and produces. The composition engine derives the coordination structure from those declarations. No one writes the chain. The chain assembles itself from the declared contracts. Adding a new agent is a registration act, not a code change to an existing workflow.

The declarative approach resolves more than just event routing. It evaluates the full dependency surface:

| Declaration | What the engine resolves |
|---|---|
| Events consumed/produced | Which agent triggers which (routing) |
| Features required | Is the data ready? (data dependency) |
| Tool capabilities needed | Is the implementation bound? (tool binding) |
| Knowledge bases needed | Is the domain knowledge available? (KB readiness) |
| Trigger conditions | When does the agent execute? (activation rules) |

An agent is only active when ALL its declarations are satisfiable — not just the event routing, but the full dependency surface.

### Use Cases

- **Plug-and-play agent addition**: Register a new agent with a manifest. The engine evaluates its declared dependencies against available resources. If all are met, the agent is active and integrated into the coordination graph immediately. If not, it's dormant with a clear gap report. No existing agents are modified in either case.
- **Dependency resolution at registration time**: Before an agent runs, the engine validates its full dependency surface — events, features, tools, knowledge bases. You know at registration time whether the agent can operate, not at runtime when it fails.
- **Automatic coordination graph**: The engine derives the event flow graph and execution order from manifest declarations. No one writes routing rules or workflow definitions. The coordination graph updates automatically when agents are added or removed.
- **Reactive lifecycle management**: When a new resource is registered (a feature computed, a tool bound, a KB indexed), the engine re-evaluates all dormant agents and activates any whose dependencies are now fully satisfied. When a resource is removed, active agents that depended on it transition back to dormant.
- **Gap reporting**: For any dormant agent, the engine reports exactly which dependencies are unresolved and what type they are (event, feature, tool, knowledge base). This is the work backlog — auto-generated from declarations.
- **Impact analysis**: Before removing a resource, the engine can preview which active agents would break. Before adding a resource, it can preview which dormant agents would activate.
- **Local validation**: Developers validate their composition logic on their machine — no cloud platform required. The same declarations that work locally are deployed to managed platforms without modification.

## Design

### Building on `AgentManifest.satisfies()`

RFC 0001 already includes a `satisfies()` method on `AgentManifest` that performs basic dependency checking for a single manifest. RFC 0002 extends this into a full composition engine that operates across multiple agents:

| `manifest.satisfies()` (RFC 0001) | `strands.composition` (this RFC) |
|---|---|
| Single manifest, single check | Multiple manifests, continuous resolution |
| Returns bool + gaps | Manages lifecycle state transitions |
| No routing | Derives event flow graph from declarations |
| Stateless | Reactive to resource changes |
| No cross-agent awareness | Knows which agents produce what others consume |

### New Module: `strands.composition`

```python
# strands/composition/__init__.py

from strands.composition.registry import CompositionRegistry
from strands.composition.resolver import BindingResolver, BindingStatus, Gap
from strands.composition.router import EventRouter, Route
from strands.composition.lifecycle import LifecycleManager, AgentState
```

### CompositionRegistry

A local catalog of agents and their manifests. Provides lookup and query capabilities.

```python
# strands/composition/registry.py

from dataclasses import dataclass, field
from strands import Agent
from strands.agent.manifest import AgentManifest


class CompositionRegistry:
    """Local registry of agents participating in a composition.
    
    Holds agent references and their manifests, enabling dependency resolution
    and route computation across the set of registered agents.
    """
    
    def __init__(self):
        self._agents: dict[str, Agent] = {}
    
    def register(self, agent: Agent) -> None:
        """Register an agent. Requires agent to have a manifest."""
        if agent.manifest is None:
            raise ValueError(f"Cannot register agent without a manifest")
        self._agents[agent.manifest.name] = agent
    
    def unregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        del self._agents[name]
    
    def get(self, name: str) -> Agent | None:
        """Get an agent by manifest name."""
        return self._agents.get(name)
    
    @property
    def agents(self) -> list[Agent]:
        """All registered agents."""
        return list(self._agents.values())
    
    @property
    def manifests(self) -> list[AgentManifest]:
        """All registered manifests."""
        return [a.manifest for a in self._agents.values()]
    
    def producers_of(self, event_type: str) -> list[str]:
        """Find agents that produce a given event type."""
        return [
            m.name for m in self.manifests
            if event_type in m.output_contract.events_produced
        ]
    
    def consumers_of(self, event_type: str) -> list[str]:
        """Find agents that consume a given event type."""
        return [
            m.name for m in self.manifests
            if event_type in m.input_contract.events_consumed
        ]
```

### BindingResolver

Evaluates a manifest against available resources and determines if the agent can activate.

```python
# strands/composition/resolver.py

from dataclasses import dataclass, field
from strands.agent.manifest import AgentManifest


@dataclass
class Gap:
    """A single unresolved dependency."""
    type: str  # "event", "feature", "tool", "knowledge_base"
    name: str
    description: str = ""


@dataclass
class BindingStatus:
    """Result of evaluating a manifest against available resources."""
    agent_name: str
    state: str  # "active" or "dormant"
    resolved_events: list[str] = field(default_factory=list)
    resolved_features: list[str] = field(default_factory=list)
    resolved_tools: list[str] = field(default_factory=list)
    resolved_knowledge_bases: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    
    def gap_report(self) -> str:
        """Human-readable gap report."""
        if not self.gaps:
            return f"{self.agent_name}: all bindings resolved (active)"
        lines = [f"{self.agent_name}: {len(self.gaps)} unresolved binding(s) (dormant)"]
        for gap in self.gaps:
            lines.append(f"  ❌ {gap.type}: {gap.name}")
        return "\n".join(lines)


class BindingResolver:
    """Evaluates agent manifests against available resources.
    
    Given a set of available events, features, tools, and knowledge bases,
    determines whether each agent's input contract can be satisfied.
    """
    
    def __init__(
        self,
        available_events: list[str] | None = None,
        available_features: list[str] | None = None,
        available_tools: list[str] | None = None,
        available_knowledge_bases: list[str] | None = None,
    ):
        self._events: set[str] = set(available_events or [])
        self._features: set[str] = set(available_features or [])
        self._tools: set[str] = set(available_tools or [])
        self._knowledge_bases: set[str] = set(available_knowledge_bases or [])
    
    def add_event(self, event: str) -> None:
        self._events.add(event)
    
    def remove_event(self, event: str) -> None:
        self._events.discard(event)
    
    def add_feature(self, feature: str) -> None:
        self._features.add(feature)
    
    def remove_feature(self, feature: str) -> None:
        self._features.discard(feature)
    
    def add_tool(self, tool: str) -> None:
        self._tools.add(tool)
    
    def remove_tool(self, tool: str) -> None:
        self._tools.discard(tool)
    
    def add_knowledge_base(self, kb: str) -> None:
        self._knowledge_bases.add(kb)
    
    def remove_knowledge_base(self, kb: str) -> None:
        self._knowledge_bases.discard(kb)
    
    def evaluate(self, manifest: AgentManifest) -> BindingStatus:
        """Evaluate a manifest against current available resources.
        
        Returns a BindingStatus indicating whether the agent can activate
        (all dependencies met) or is dormant (gaps exist).
        """
        gaps = []
        resolved_events = []
        resolved_features = []
        resolved_tools = []
        resolved_kbs = []
        
        # Check events
        for event in manifest.input_contract.events_consumed:
            if event in self._events:
                resolved_events.append(event)
            else:
                gaps.append(Gap(type="event", name=event))
        
        # Check features
        for feature in manifest.input_contract.features_required:
            if feature in self._features:
                resolved_features.append(feature)
            else:
                gaps.append(Gap(type="feature", name=feature))
        
        # Check tools
        for tool in manifest.input_contract.tool_capabilities:
            if tool in self._tools:
                resolved_tools.append(tool)
            else:
                gaps.append(Gap(type="tool", name=tool))
        
        # Check knowledge bases
        for kb in manifest.input_contract.knowledge_bases:
            if kb in self._knowledge_bases:
                resolved_kbs.append(kb)
            else:
                gaps.append(Gap(type="knowledge_base", name=kb))
        
        return BindingStatus(
            agent_name=manifest.name,
            state="active" if not gaps else "dormant",
            resolved_events=resolved_events,
            resolved_features=resolved_features,
            resolved_tools=resolved_tools,
            resolved_knowledge_bases=resolved_kbs,
            gaps=gaps,
        )
    
    def evaluate_all(self, manifests: list[AgentManifest]) -> list[BindingStatus]:
        """Evaluate multiple manifests. Returns status for each."""
        return [self.evaluate(m) for m in manifests]
    
    def what_would_activate(self, resource_type: str, resource_name: str, manifests: list[AgentManifest]) -> list[str]:
        """Preview: which dormant agents would activate if this resource were added?
        
        Does not mutate state. Useful for impact analysis.
        """
        # Temporarily add the resource
        original = getattr(self, f"_{resource_type}s").copy()
        getattr(self, f"_{resource_type}s").add(resource_name)
        
        would_activate = []
        for manifest in manifests:
            status = self.evaluate(manifest)
            if status.state == "active":
                # Check if it was dormant before
                getattr(self, f"_{resource_type}s").discard(resource_name)
                prev_status = self.evaluate(manifest)
                getattr(self, f"_{resource_type}s").add(resource_name)
                if prev_status.state == "dormant":
                    would_activate.append(manifest.name)
        
        # Restore
        setattr(self, f"_{resource_type}s", original)
        return would_activate
```

### EventRouter

Computes the event flow graph from manifest declarations.

```python
# strands/composition/router.py

from dataclasses import dataclass
from strands.composition.registry import CompositionRegistry


@dataclass(frozen=True)
class Route:
    """A resolved event route: source agent produces event, target agents consume it."""
    event: str
    source: str  # agent name that produces this event
    targets: list[str]  # agent names that consume this event
    
    def __eq__(self, other):
        if not isinstance(other, Route):
            return False
        return (self.event == other.event and 
                self.source == other.source and 
                set(self.targets) == set(other.targets))
    
    def __hash__(self):
        return hash((self.event, self.source, frozenset(self.targets)))


class EventRouter:
    """Computes event routing from manifest declarations.
    
    Given a registry of agents, determines how events flow between them
    based on output_contract.events_produced and input_contract.events_consumed.
    No manual wiring needed — routes emerge from contracts.
    """
    
    def __init__(self, registry: CompositionRegistry):
        self._registry = registry
    
    def resolve_routes(self) -> list[Route]:
        """Compute all event routes across registered agents.
        
        For each event type produced by any agent, find all agents that
        consume it and create a route.
        """
        routes = []
        
        # Collect all produced event types and their producers
        produced: dict[str, list[str]] = {}
        for manifest in self._registry.manifests:
            for event in manifest.output_contract.events_produced:
                produced.setdefault(event, []).append(manifest.name)
        
        # For each produced event, find consumers
        for event_type, producers in produced.items():
            consumers = self._registry.consumers_of(event_type)
            if consumers:
                for producer in producers:
                    # Don't route to self
                    targets = [c for c in consumers if c != producer]
                    if targets:
                        routes.append(Route(
                            event=event_type,
                            source=producer,
                            targets=targets,
                        ))
        
        return routes
    
    def routes_from(self, agent_name: str) -> list[Route]:
        """Get all routes originating from a specific agent."""
        all_routes = self.resolve_routes()
        return [r for r in all_routes if r.source == agent_name]
    
    def routes_to(self, agent_name: str) -> list[Route]:
        """Get all routes targeting a specific agent."""
        all_routes = self.resolve_routes()
        return [r for r in all_routes if agent_name in r.targets]
    
    def dependency_order(self) -> list[list[str]]:
        """Topological sort of agents by event dependencies.
        
        Returns layers: agents in layer 0 have no event dependencies,
        agents in layer 1 depend only on layer 0 outputs, etc.
        Useful for understanding execution order.
        """
        manifests = {m.name: m for m in self._registry.manifests}
        
        # Build dependency graph
        deps: dict[str, set[str]] = {name: set() for name in manifests}
        for manifest in manifests.values():
            for event in manifest.input_contract.events_consumed:
                producers = self._registry.producers_of(event)
                for p in producers:
                    if p != manifest.name:
                        deps[manifest.name].add(p)
        
        # Topological sort (Kahn's algorithm)
        layers = []
        remaining = dict(deps)
        
        while remaining:
            # Find agents with no unresolved dependencies
            layer = [n for n, d in remaining.items() if not d]
            if not layer:
                # Cycle detected — return what we have
                layer = list(remaining.keys())
                layers.append(layer)
                break
            layers.append(layer)
            # Remove resolved agents from dependencies
            for n in layer:
                del remaining[n]
            for d in remaining.values():
                d -= set(layer)
        
        return layers
    
    def visualize(self) -> str:
        """ASCII visualization of the event flow graph."""
        routes = self.resolve_routes()
        if not routes:
            return "(no routes)"
        
        lines = []
        for route in routes:
            targets_str = ", ".join(route.targets)
            lines.append(f"  {route.source} ──[{route.event}]──▶ {targets_str}")
        
        return "Event Flow:\n" + "\n".join(sorted(lines))
```

### LifecycleManager

Manages dormant/active transitions reactively as resources change.

```python
# strands/composition/lifecycle.py

from dataclasses import dataclass, field
from datetime import datetime
from strands.composition.registry import CompositionRegistry
from strands.composition.resolver import BindingResolver, BindingStatus


@dataclass
class StateTransition:
    """Record of an agent state change."""
    agent_name: str
    from_state: str
    to_state: str
    timestamp: datetime
    reason: str  # what resource triggered the transition


@dataclass
class AgentState:
    """Current state of an agent in the composition."""
    name: str
    state: str  # "active" or "dormant"
    binding_status: BindingStatus


class LifecycleManager:
    """Manages agent lifecycle based on resource availability.
    
    When resources are added or removed, re-evaluates all agents and
    transitions them between dormant and active states.
    """
    
    def __init__(self, registry: CompositionRegistry, resolver: BindingResolver):
        self._registry = registry
        self._resolver = resolver
        self._states: dict[str, str] = {}  # agent_name → state
        self._history: list[StateTransition] = []
        
        # Initial evaluation
        self._evaluate_all()
    
    def _evaluate_all(self) -> list[StateTransition]:
        """Re-evaluate all agents and return any state transitions."""
        transitions = []
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
            
            self._states[manifest.name] = new_state
        
        return transitions
    
    def add_resource(self, resource_type: str, name: str) -> list[StateTransition]:
        """Add a resource and return any resulting state transitions.
        
        resource_type: "event", "feature", "tool", or "knowledge_base"
        """
        add_method = getattr(self._resolver, f"add_{resource_type}")
        add_method(name)
        
        transitions = self._evaluate_all()
        for t in transitions:
            t.reason = f"{resource_type} '{name}' added"
        
        return transitions
    
    def remove_resource(self, resource_type: str, name: str) -> list[StateTransition]:
        """Remove a resource and return any resulting state transitions."""
        remove_method = getattr(self._resolver, f"remove_{resource_type}")
        remove_method(name)
        
        transitions = self._evaluate_all()
        for t in transitions:
            t.reason = f"{resource_type} '{name}' removed"
        
        return transitions
    
    def get_state(self, agent_name: str) -> AgentState:
        """Get current state of an agent."""
        manifest = self._registry.get(agent_name).manifest
        status = self._resolver.evaluate(manifest)
        return AgentState(name=agent_name, state=status.state, binding_status=status)
    
    @property
    def active_agents(self) -> list[str]:
        return [name for name, state in self._states.items() if state == "active"]
    
    @property
    def dormant_agents(self) -> list[str]:
        return [name for name, state in self._states.items() if state == "dormant"]
    
    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)
```

## Usage Examples

### Example 1: Validate a composition before deployment

```python
from strands import Agent
from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger
from strands.composition import CompositionRegistry, BindingResolver, EventRouter

# Define agents
rca_agent = Agent(
    manifest=AgentManifest(
        name="rca_agent",
        version="1.0.0",
        input_contract=InputContract(events_consumed=["AlarmEvent"]),
        output_contract=OutputContract(events_produced=["FaultEvent"]),
        trigger=Trigger(type="event", condition="AlarmEvent"),
    ),
    ...
)

impact_agent = Agent(
    manifest=AgentManifest(
        name="impact_agent",
        version="1.0.0",
        input_contract=InputContract(
            events_consumed=["FaultEvent"],
            features_required=["service_mapping"],
        ),
        output_contract=OutputContract(events_produced=["ImpactEvent"]),
        trigger=Trigger(type="event", condition="FaultEvent"),
    ),
    ...
)

# Check composition
registry = CompositionRegistry()
registry.register(rca_agent)
registry.register(impact_agent)

resolver = BindingResolver(
    available_events=["AlarmEvent"],
    available_features=[],  # service_mapping not yet available
    available_tools=[],
)

for manifest in registry.manifests:
    status = resolver.evaluate(manifest)
    print(status.gap_report())

# Output:
# rca_agent: all bindings resolved (active)
# impact_agent: 1 unresolved binding(s) (dormant)
#   ❌ feature: service_mapping

# Visualize the routes (only active agents would execute)
router = EventRouter(registry)
print(router.visualize())

# Output:
# Event Flow:
#   rca_agent ──[FaultEvent]──▶ impact_agent
```

### Example 2: Simulate resource arrival

```python
from strands.composition import LifecycleManager

manager = LifecycleManager(registry, resolver)

print(manager.active_agents)   # ["rca_agent"]
print(manager.dormant_agents)  # ["impact_agent"]

# Feature arrives
transitions = manager.add_resource("feature", "service_mapping")

for t in transitions:
    print(f"{t.agent_name}: {t.from_state} → {t.to_state} ({t.reason})")

# Output:
# impact_agent: dormant → active (feature 'service_mapping' added)

print(manager.active_agents)   # ["rca_agent", "impact_agent"]
```

### Example 3: Impact analysis

```python
# What breaks if I remove this feature?
transitions = manager.remove_resource("feature", "service_mapping")

for t in transitions:
    print(f"{t.agent_name}: {t.from_state} → {t.to_state} ({t.reason})")

# Output:
# impact_agent: active → dormant (feature 'service_mapping' removed)
```

### Example 4: Execution order

```python
router = EventRouter(registry)
layers = router.dependency_order()

# layers = [
#   ["rca_agent"],       # layer 0: no event dependencies (triggered by external AlarmEvent)
#   ["impact_agent"],    # layer 1: depends on rca_agent's FaultEvent
# ]
```

## When to Use

Use `strands.composition` when:
- You have multiple agents with declared dependencies (manifests)
- You want to validate that a multi-agent system's full dependency surface is satisfiable — not just events, but features, tools, and knowledge bases
- You want the coordination structure derived from declarations, not written as imperative code
- You need lifecycle management (dormant until all dependencies are met, not just event subscriptions)
- You want to understand the impact of adding or removing any resource type
- You want to test composition logic locally before deploying to a managed platform
- You want plug-and-play extensibility: new agents register and activate without modifying existing agents

Do NOT use `strands.composition` when:
- You have a single agent with no dependencies on other agents
- Your agents communicate through direct function calls (not declarative contracts)
- You prefer explicit imperative orchestration (Step Functions, DAG definitions) and accept the maintenance cost of updating the DAG when agents change
- Your agents have no formal input/output contracts

## Relationship to Existing Multi-Agent Patterns

Strands already provides two multi-agent coordination patterns:

- **Graph** (`strands.multiagent.graph`): Deterministic DAG execution. The developer explicitly defines nodes, edges, and entry points. The structure is imperative — you write it, you maintain it. Best for: fixed workflows where the execution order is known at design time.

- **Swarm** (`strands.multiagent.swarm`): Self-organizing agents with shared context. Agents hand off to each other at runtime via a `handoff_to_agent` tool. The structure is emergent at runtime — agents decide who to call based on the task. Best for: collaborative problem-solving where the right agent depends on the conversation.

**Composition** (this RFC) adds a third pattern:

- **Composition** (`strands.composition`): Declarative coordination. Agents declare what they need and produce via manifests. The structure is derived from declarations — no one writes it, no one maintains it. Adding an agent is registering a manifest. Best for: systems where agents are developed independently, deployed incrementally, and must compose without modifying each other.

| | Graph | Swarm | Composition |
|---|---|---|---|
| Topology defined by | Developer (imperative) | Agents at runtime | Manifest declarations |
| Adding a new agent | Edit graph code | Add to swarm node list | Register manifest |
| Dependency validation | None (runtime failure) | None (runtime failure) | At registration time |
| Execution model | Batch (run full DAG) | Interactive (handoff loop) | Event-driven (trigger on input) |
| Best for | Fixed pipelines | Collaborative tasks | Plug-and-play extensibility |

These patterns are complementary. A Graph node could be an agent with a manifest. A Swarm could use manifests for discovery. Composition provides the dependency resolution layer that neither Graph nor Swarm has today.

---

The composition module is pure resolution logic — it computes binding status, routes, and lifecycle transitions. It does not execute agents, deliver events, or persist state. Those responsibilities belong to the execution environment:

- **Local development**: Resolution results are informational. Developer uses them to validate and understand the composition.
- **Test harness**: A test runner reads routes and invokes agents in dependency order.
- **Managed platform (e.g., AgentCore)**: The platform reads the same manifests, uses the same declarative resolution logic, and implements the coordination via managed services (EventBridge for routing, Registry for lifecycle, Policy for contract enforcement).

The composition module defines the **declarative protocol**. Managed platforms provide the **production execution**.

## Backward Compatibility

- New module, no changes to existing SDK APIs
- Requires RFC 0001 (AgentManifest) to be merged first
- No new required dependencies (pure Python, standard library only)
- Agents without manifests cannot participate in composition (by design)
- If this is considered too opinionated for core, it can live as a standalone package (`strands-agents/composition`) with no changes to the implementation

## Alternatives Considered

1. **Orchestrator pattern (imperative DAG)**: Rejected. Requires someone to write and maintain the workflow definition. Adding an agent means modifying the orchestrator. The declarative approach is additive — new agents plug in without modifying existing definitions.

2. **Pub/Sub only (no dependency validation)**: Rejected. Pub/Sub solves routing but not readiness. An agent can subscribe to a topic but still fail because a feature isn't available or a tool isn't bound. Declarative composition validates the full dependency surface, not just event subscriptions.

3. **Blackboard pattern (shared state)**: Rejected. No explicit contracts, no dependency validation, no lifecycle management. Agents silently fail when expected state isn't present.

4. **Convention-based routing (match by name)**: Rejected. Fragile. Explicit declarative contracts are more reliable than naming conventions.

5. **Build into Agent class directly**: Rejected. Composition is a concern of the system, not the individual agent. Keeping it in a separate module maintains separation of concerns — agents declare, the composition engine resolves.

## Implementation Plan

1. Merge RFC 0001 (AgentManifest) — prerequisite
2. Add `src/strands/composition/__init__.py` with module exports
3. Add `src/strands/composition/registry.py` — CompositionRegistry
4. Add `src/strands/composition/resolver.py` — BindingResolver, BindingStatus, Gap
5. Add `src/strands/composition/router.py` — EventRouter, Route
6. Add `src/strands/composition/lifecycle.py` — LifecycleManager, AgentState, StateTransition
7. Add unit tests for all components
8. Add integration test: multi-agent composition scenario end-to-end (in-process)
9. Add documentation page with examples
10. Export from `strands.composition`
