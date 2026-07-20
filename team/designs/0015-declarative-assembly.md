# Declarative Agent Assembly

**Status**: Proposed

**Date**: 2026-06-10

**Issue**: [#3355](https://github.com/strands-agents/harness-sdk/issues/3355)

**Related**:
- [#2270: AgentManifest implementation](https://github.com/strands-agents/harness-sdk/pull/2270)
- [#2273: Assembly engine implementation](https://github.com/strands-agents/harness-sdk/pull/2273)

---

## Problem

Multi-agent systems in Strands have no way to validate dependencies before execution. An agent that requires a specific tool, data source, or upstream event discovers at runtime that the dependency is missing — after it has already been invoked and failed. There is no mechanism to declare what an agent needs, what it produces, or whether its requirements are satisfiable in the current environment.

This creates three concrete problems as the number of agents grows:

1. **No dependency declaration.** A Graph node can fail because the tool it needs is not configured or the data source it expects does not exist. A Swarm agent can receive a handoff it cannot handle because nothing validates its requirements before invocation. In both cases, the failure mode is a runtime error.

2. **No composability contract.** Two agents cannot be connected without custom wiring code. If Agent A produces an event that Agent B needs, a developer must write the glue — there is no way for the framework to derive this from declarations. Every new agent means modifying existing orchestration code.

3. **No lifecycle management.** An agent cannot be registered in advance and activated later when its dependencies become available. It is either deployed and expected to work, or not deployed at all. There is no dormant state.

---

## Current State

Strands provides two multi-agent coordination patterns:

**Graph** (`strands.multiagent.graph`): The developer defines nodes, edges, and entry points using `GraphBuilder`. Execution follows the graph structure deterministically. Adding an agent means calling `add_node()` and `add_edge()` — modifying the graph definition. There is no pre-validation that a node's agent can actually operate (tools bound, data available). If a node fails, the graph fails.

**Swarm** (`strands.multiagent.swarm`): Agents hand off to each other at runtime via a `handoff_to_agent` tool. The LLM decides who to call based on agent names and descriptions. There is no pre-validation that the receiving agent can handle the request. If an agent lacks a required tool or data source, it fails mid-conversation.

Both patterns share the same gap: **dependency satisfaction is unchecked until execution**. This is acceptable for small systems (3–5 agents, one team) where the developer holds the full picture. It becomes increasingly costly as agent count grows:

- At 8–12 agents, graph modifications require understanding all existing edges. Swarm handoff chains become unpredictable.
- At 15+ agents across multiple teams and domains, no single developer holds the full dependency picture. Adding an agent requires cross-team coordination to understand what already exists and what needs to be wired.

The cost is not just runtime failures. It is the inability to answer basic questions about the system before running it: Can this agent operate here? What depends on it? What breaks if I remove this tool?

---

## Goals and Non-Goals

**Goals:**
- Enable agents to declare their input requirements and output types as structured metadata
- Provide a resolution engine that evaluates declarations against available resources and reports gaps
- Derive coordination structure (event flow, execution order) from declarations without imperative wiring
- Support lifecycle transitions (dormant ↔ active) as resources arrive or depart
- Complement Graph and Swarm — work alongside them, not replace them
- Pure Python, no infrastructure dependencies, testable locally

**Non-Goals:**
- Agent execution — the module does not invoke agents or deliver events
- State persistence — the module does not persist assembly state
- Cloud integration — the module does not require any managed service
- Replacing Graph or Swarm — they remain the right tools for fixed pipelines and collaborative tasks

---

## Proposal

### Recommended: AgentManifest + Assembly Engine

Two tightly coupled components that together enable declarative multi-agent coordination.

#### Part 1: AgentManifest

An optional `manifest` parameter on the `Agent` class. Purely metadata — does not change execution behavior.

```python
@dataclass
class InputContract:
    features_required: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    events_consumed: list[str] = field(default_factory=list)
    tool_capabilities: list[str] = field(default_factory=list)
    knowledge_bases: list[str] = field(default_factory=list)

@dataclass
class OutputContract:
    events_produced: list[str] = field(default_factory=list)
    features_produced: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)

@dataclass
class Trigger:
    type: str  # "event", "schedule", "on_demand"
    condition: Optional[str] = None

@dataclass
class AgentManifest:
    name: str
    version: str
    domain: Optional[str] = None
    description: Optional[str] = None
    input_contract: InputContract = field(default_factory=InputContract)
    output_contract: OutputContract = field(default_factory=OutputContract)
    trigger: Optional[Trigger] = None
    scenario_types: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "AgentManifest": ...
    @classmethod
    def from_file(cls, path: str) -> "AgentManifest": ...
    def satisfies(self, available_events, available_features, available_tools, available_kbs) -> tuple[bool, list[str]]: ...
```

The `Agent` class gains one optional parameter:

```python
class Agent(AgentBase):
    def __init__(self, ..., manifest: AgentManifest | None = None, ...):
```

The manifest is never sent to the model. It has zero runtime cost on agent execution.

#### Part 2: Assembly Engine (`strands.assembly`)

Four components that read manifests and derive coordination:

| Component | Responsibility |
|-----------|---------------|
| `AssemblyRegistry` | Local catalog of agents. Lookup by name, query producers/consumers of event types. |
| `BindingResolver` | Evaluates a manifest against available resources. Returns binding status (active/dormant) and a list of unresolved gaps. |
| `EventRouter` | Derives the event flow graph from manifest declarations. Computes topological execution order. |
| `LifecycleManager` | Manages dormant ↔ active transitions reactively as resources are added or removed. |

The module is pure resolution logic. It computes what *could* run and how things *would* connect. It does not execute agents, deliver events, or persist state.

**Pros:**
- Pre-deployment validation — dependencies are checked at registration time, not at runtime
- Additive coordination — adding an agent is registering a manifest, not modifying existing code
- Gap reporting — for any dormant agent, the engine reports exactly which dependencies are unresolved
- Impact analysis — before removing a resource, the engine previews which agents would break
- Works with Graph and Swarm — manifests on Graph nodes enable pre-run validation; manifests on Swarm agents enable capability-based handoff filtering
- No new dependencies — pure Python, standard library only

**Cons:**
- Agents must define a manifest to participate — upfront effort for each agent
- The manifest schema is a commitment — changes affect all tooling that reads it
- Developers learn a new concept alongside Graph and Swarm
- Declarative assembly is a different mental model from imperative wiring — may not suit teams that prefer explicit control

### Alternative: Extend Graph with dependency validation

Add a `requires` parameter to `GraphBuilder.add_node()` that declares what each node needs. The graph validates all nodes before execution.

```python
builder.add_node(agent, "rca", requires={"tools": ["alarm_query"], "features": ["topology"]})
```

**Pros:**
- Simpler — no new module, no new concept
- Lives entirely within the existing Graph pattern

**Cons:**
- Graph-only — Swarm gets nothing, standalone agents get nothing
- No cross-agent assembly — no way to derive routing from declarations
- No lifecycle management — no dormant state, no reactive activation
- Couples dependency metadata to the graph definition rather than the agent itself
- Cannot be used for registry/catalog integration outside Graph

### Alternative: Convention-based routing

Match agents by naming convention (e.g., agent producing `X_event` automatically connects to agents consuming `X_event`).

**Pros:**
- Zero configuration — works by naming alone

**Cons:**
- Fragile — a rename breaks routing silently
- No validation — no way to check if a consumer's full dependency surface is satisfied
- No lifecycle management
- Implicit behavior makes the system harder to reason about

---

## Developer Experience

### Registering agents and validating assembly

```python
from strands import Agent
from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger
from strands.assembly import AssemblyRegistry, BindingResolver, EventRouter

# Each agent declares its contract
alarm_agent = Agent(
    manifest=AgentManifest(
        name="alarm_search",
        version="1.0.0",
        input_contract=InputContract(events_consumed=["AlarmQuery"]),
        output_contract=OutputContract(events_produced=["AlarmData"]),
        trigger=Trigger(type="event", condition="AlarmQuery"),
    ),
    system_prompt="...",
    tools=[...],
)

rca_agent = Agent(
    manifest=AgentManifest(
        name="root_cause_analyzer",
        version="1.0.0",
        input_contract=InputContract(
            events_consumed=["AlarmData"],
            features_required=["device_topology"],
            tool_capabilities=["graph_traversal"],
        ),
        output_contract=OutputContract(events_produced=["RCAResult"]),
        trigger=Trigger(type="event", condition="AlarmData"),
    ),
    system_prompt="...",
    tools=[...],
)

# Register
registry = AssemblyRegistry()
registry.register(alarm_agent)
registry.register(rca_agent)

# Validate against available resources
resolver = BindingResolver(
    available_events=["AlarmQuery"],
    available_features=[],  # device_topology not yet available
    available_tools=["graph_traversal"],
)

for manifest in registry.manifests:
    status = resolver.evaluate(manifest)
    print(status.gap_report())

# Output:
# alarm_search: all bindings resolved (active)
# root_cause_analyzer: 1 unresolved binding(s) (dormant)
#   ❌ feature: device_topology
```

### Deriving event routes

```python
router = EventRouter(registry)
print(router.visualize())
# Event Flow:
#   alarm_search ──[AlarmData]──▶ root_cause_analyzer

layers = router.dependency_order()
# [["alarm_search"], ["root_cause_analyzer"]]
```

### Reactive lifecycle

```python
from strands.assembly import LifecycleManager

manager = LifecycleManager(registry, resolver)
print(manager.active_agents)   # ["alarm_search"]
print(manager.dormant_agents)  # ["root_cause_analyzer"]

# Feature arrives
transitions = manager.add_resource("feature", "device_topology")
for t in transitions:
    print(f"{t.agent_name}: {t.from_state} → {t.to_state} ({t.reason})")
# root_cause_analyzer: dormant → active (feature 'device_topology' added)
```

### Adding an agent without modifying existing code

```python
# New agent — consumes RCAResult, produces ImpactReport
impact_agent = Agent(
    manifest=AgentManifest(
        name="service_impact",
        version="1.0.0",
        input_contract=InputContract(
            events_consumed=["RCAResult"],
            features_required=["service_catalog"],
        ),
        output_contract=OutputContract(events_produced=["ImpactReport"]),
        trigger=Trigger(type="event", condition="RCAResult"),
    ),
    system_prompt="...",
    tools=[...],
)

# Register — no existing agents modified
registry.register(impact_agent)

# Routes update automatically
router = EventRouter(registry)
print(router.visualize())
# Event Flow:
#   alarm_search ──[AlarmData]──▶ root_cause_analyzer
#   root_cause_analyzer ──[RCAResult]──▶ service_impact
```

### Using manifests with Graph (pre-run validation)

```python
from strands.multiagent import GraphBuilder

builder = GraphBuilder()
builder.add_node(alarm_agent, "alarm")
builder.add_node(rca_agent, "rca")
builder.add_edge("alarm", "rca")
graph = builder.build()

# Before running, validate that all nodes can operate
for node_id, node in builder.nodes.items():
    if node.executor.manifest:
        status = resolver.evaluate(node.executor.manifest)
        if status.state == "dormant":
            raise RuntimeError(f"Node '{node_id}' cannot operate: {status.gap_report()}")

result = graph("Investigate the alarm")
```

### Using manifests with Swarm (capability filtering)

```python
# When selecting handoff targets, filter by actual capability
def capable_agents_for_event(swarm_agents, event_type, resolver):
    """Return agents that consume this event and have all dependencies met."""
    return [
        agent for agent in swarm_agents
        if agent.manifest
        and event_type in agent.manifest.input_contract.events_consumed
        and resolver.evaluate(agent.manifest).state == "active"
    ]
```

### File-based manifest

```json
{
  "name": "root_cause_analyzer",
  "version": "1.0.0",
  "domain": "network_operations",
  "input_contract": {
    "events_consumed": ["AlarmData"],
    "features_required": ["device_topology"],
    "tool_capabilities": ["graph_traversal"]
  },
  "output_contract": {
    "events_produced": ["RCAResult"]
  },
  "trigger": {"type": "event", "condition": "AlarmData"}
}
```

```python
agent = Agent(
    manifest=AgentManifest.from_file("manifest.json"),
    system_prompt="...",
    tools=[...],
)
```

---

## How It Integrates with Existing Features

| Feature | Relationship |
|---------|-------------|
| `Agent` class | One new optional parameter (`manifest`). No other changes. |
| `Graph` | Graph nodes can carry manifests. Enables pre-run dependency validation. |
| `Swarm` | Swarm agents can carry manifests. Enables capability-based handoff filtering. |
| `Hooks` | Not affected. Manifests are metadata, not execution logic. |
| `Tools / MCP` | Tool capabilities in manifests are abstract names (e.g., `data_query`). Binding to concrete tools is external to the SDK. |
| `A2A` | Manifests could serve as capability advertisements for A2A agent discovery (future work). |
| `Sessions` | Not affected. Assembly state is independent of session state. |
| `Plugins` | Not affected. Assembly is a resolution engine, not a lifecycle hook. |

---

## Consequences

### What becomes easier

- Knowing at registration time whether an agent can operate in a given environment
- Adding agents to a multi-agent system without modifying existing orchestration code
- Understanding the coordination graph from declarations alone
- Answering "what breaks if I remove this resource?" before removing it
- CI/CD validation of agent dependencies before deployment
- Managing agent lifecycle as data sources or tools arrive or depart

### What becomes harder or requires attention

- Agents that want to participate in assembly must define a manifest. This is a small upfront cost per agent.
- The manifest schema becomes a contract — changes to it affect all consuming tooling and must be versioned carefully.
- Developers must learn the manifest concept and understand when to use Assembly versus Graph or Swarm.
- Abstract tool capabilities (declared in manifests) must be mapped to concrete tools by the runtime environment. The SDK does not enforce this mapping.

### Backward compatibility

- `manifest` parameter is optional, defaults to `None`
- Existing agents without manifests work exactly as before
- No breaking changes to any existing API
- No new required dependencies (pure Python, standard library only)
- `strands.assembly` is a new module — no modifications to existing modules

---

## Implementation

Complete. Both components are implemented with 55 passing tests (24 for manifest, 31 for assembly).

### PRs

| PR | Component | Status | Description |
|---|---|---|---|
| [#2270](https://github.com/strands-agents/harness-sdk/pull/2270) | AgentManifest | Closed — see note below | `manifest.py` + `Agent.__init__` integration + 24 unit tests |
| [#2273](https://github.com/strands-agents/harness-sdk/pull/2273) | Assembly Engine | Open (review requested) | `strands.assembly` module (registry, resolver, router, lifecycle) + 31 unit tests |

### Note on #2270

PR #2270 was closed automatically by the `autoclose in 7 days` bot. A maintainer had requested a feature request issue to explain the use case — we missed the comment within the 7-day window. The requested issue has now been filed as [#3355](https://github.com/strands-agents/harness-sdk/issues/3355), which is this design document's companion. We are requesting that #2270 be reopened so both PRs can be reviewed together.

### Module structure

```
strands/agent/manifest.py          # AgentManifest, InputContract, OutputContract, Trigger
strands/assembly/__init__.py       # Module exports
strands/assembly/registry.py       # AssemblyRegistry
strands/assembly/resolver.py       # BindingResolver, BindingStatus, Gap
strands/assembly/router.py         # EventRouter, Route
strands/assembly/lifecycle.py      # LifecycleManager, AgentState, StateTransition
```
