# RFC 0001: Agent Manifest — Declarative Capability Profile

## Summary

Add an optional `AgentManifest` to the `Agent` class that formally declares an agent's input requirements, output types, trigger conditions, and tool capabilities. This enables platforms to manage agent lifecycle, resolve dependencies automatically, and compose agents into chains without custom wiring.

## Motivation

### Problem

Today, a Strands agent is defined by its runtime configuration: model, tools, system prompt. There is no way to declare what an agent *needs* (data sources, features, upstream event types) or what it *produces* (output schemas, downstream events) without inspecting its code or documentation. This means every multi-agent deployment requires manual dependency tracking, custom wiring between agents, and runtime failures as the only signal that something is missing.

This creates three problems in production multi-agent systems:

1. **No dependency declaration**: A platform cannot determine whether an agent's required inputs are available without running it and observing failures.
2. **No composability contract**: Two agents cannot be chained without custom integration code because neither declares its input/output schema.
3. **No lifecycle management**: An agent cannot be registered in advance and activated later when its dependencies become available.

### Real-World Example

In a multi-agent data processing platform, an Enrichment Agent needs two inputs: a `RawDataEvent` from an Ingestion Agent and a `customer_profile` feature from a Feature Store. Today, this agent must be deployed and configured manually after both dependencies exist. With a manifest, the agent can be registered as dormant and activate automatically when both inputs become available — no redeployment, no manual wiring.

### Use Cases

- **Multi-agent platforms**: Agents declare what they need; the platform resolves bindings and activates agents when dependencies are met.
- **Agent registries and catalogs**: Manifests serve as the catalog entry for agent discovery — what does this agent do, what does it need, what does it produce.
- **Scenario and simulation orchestration**: A platform determines which agents to invoke for a given scenario by matching available data against agent manifests.
- **Tool capability abstraction**: An agent declares it needs a `data_query` capability, not a specific PostgreSQL tool. The platform resolves the concrete binding per deployment environment.
- **CI/CD for agents**: Manifests enable automated validation that an agent's declared dependencies are satisfiable before deployment.

## Design

### New Classes

```python
# src/strands/agent/manifest.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InputContract:
    """Declares what an agent requires to execute."""
    features_required: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    events_consumed: list[str] = field(default_factory=list)
    tool_capabilities: list[str] = field(default_factory=list)
    knowledge_bases: list[str] = field(default_factory=list)

@dataclass
class OutputContract:
    """Declares what an agent produces."""
    events_produced: list[str] = field(default_factory=list)
    features_produced: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)

@dataclass  
class Trigger:
    """Declares what activates this agent."""
    type: str  # "event", "schedule", "on_demand"
    condition: Optional[str] = None  # event type filter or cron expression

@dataclass
class AgentManifest:
    """Declarative capability profile for an agent.
    
    A manifest describes what an agent needs (inputs), what it produces (outputs),
    what activates it (trigger), and what scenarios it supports. It is purely
    declarative metadata — it does not change agent execution behavior. External
    systems (registries, orchestrators, platforms) read manifests to manage agent
    lifecycle and composition.
    """
    name: str
    version: str
    domain: Optional[str] = None
    description: Optional[str] = None
    
    input_contract: InputContract = field(default_factory=InputContract)
    output_contract: OutputContract = field(default_factory=OutputContract)
    trigger: Optional[Trigger] = None
    scenario_types: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize manifest to dictionary for registry registration."""
        ...
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentManifest":
        """Deserialize manifest from dictionary."""
        ...
    
    @classmethod
    def from_file(cls, path: str) -> "AgentManifest":
        """Load manifest from JSON or YAML file."""
        ...
```

### Agent Class Changes

```python
# In Agent.__init__, add optional manifest parameter:

class Agent(AgentBase):
    def __init__(
        self,
        ...,
        manifest: AgentManifest | None = None,  # NEW
        ...
    ):
```

The manifest is purely declarative. It does not change agent execution behavior. It is metadata that external systems (registries, orchestrators, platforms) can read to manage the agent's lifecycle.

### Usage Examples

#### Basic: Agent with manifest

```python
from strands import Agent, tool
from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger

@tool
def analyze_data(query: str) -> dict:
    """Run analysis on structured data."""
    ...

agent = Agent(
    manifest=AgentManifest(
        name="data_analysis_agent",
        version="1.0.0",
        domain="analytics",
        description="Analyzes structured datasets and produces insight reports",
        input_contract=InputContract(
            features_required=["dataset_schema", "recent_metrics"],
            events_consumed=["DataReadyEvent"],
            tool_capabilities=["data_query", "chart_generation"],
        ),
        output_contract=OutputContract(
            events_produced=["InsightEvent"],
            artifacts_produced=["analysis_report"],
        ),
        trigger=Trigger(type="event", condition="DataReadyEvent"),
    ),
    system_prompt="You are a data analysis agent...",
    tools=[analyze_data],
)

# Access manifest programmatically
print(agent.manifest.input_contract.features_required)
# ['dataset_schema', 'recent_metrics']

# Serialize for registry registration
manifest_dict = agent.manifest.to_dict()
```

#### Multi-agent composition via manifests

```python
# Agent A produces events that Agent B consumes
# No direct wiring needed — a platform reads both manifests and connects them

ingestion_agent = Agent(
    manifest=AgentManifest(
        name="ingestion_agent",
        version="1.0.0",
        output_contract=OutputContract(events_produced=["RawDataEvent"]),
    ),
    ...
)

enrichment_agent = Agent(
    manifest=AgentManifest(
        name="enrichment_agent",
        version="1.0.0",
        input_contract=InputContract(events_consumed=["RawDataEvent"]),
        output_contract=OutputContract(events_produced=["EnrichedDataEvent"]),
        trigger=Trigger(type="event", condition="RawDataEvent"),
    ),
    ...
)

# A platform reads both manifests and knows:
# ingestion_agent.output → RawDataEvent → enrichment_agent.input
# No custom integration code needed.
```

#### File-based manifest

```json
{
  "name": "enrichment_agent",
  "version": "1.0.0",
  "domain": "data_pipeline",
  "input_contract": {
    "features_required": ["customer_profile"],
    "events_consumed": ["RawDataEvent"],
    "tool_capabilities": ["data_query", "enrichment_api"]
  },
  "output_contract": {
    "events_produced": ["EnrichedDataEvent"]
  },
  "trigger": {"type": "event", "condition": "RawDataEvent"}
}
```

```python
from strands.agent.manifest import AgentManifest

agent = Agent(
    manifest=AgentManifest.from_file("manifest.json"),
    system_prompt="...",
    tools=[...],
)
```

## When to Use

Use `AgentManifest` when:
- Your agent is part of a multi-agent system where agents need to discover each other's capabilities
- You want to register agents in a catalog or registry with formal input/output declarations
- You need lifecycle management (agents that activate when dependencies are met)
- You want to enable orchestration where a platform selects agents based on their declared capabilities
- You want to abstract tool requirements from concrete implementations (capability names vs specific tools)
- You want CI/CD validation that an agent's dependencies are satisfiable before deployment

Do NOT use `AgentManifest` when:
- You have a simple single-agent application
- Your agent's inputs and outputs are ad-hoc and not formalized
- You don't need external lifecycle management or registry integration

## Backward Compatibility

- `manifest` parameter is optional and defaults to `None`
- Existing agents without manifests work exactly as before
- No breaking changes to any existing API
- No new required dependencies

## Alternatives Considered

1. **Separate manifest file only (no SDK integration)**: Rejected because it creates drift between the manifest and the actual agent configuration. Keeping the manifest in the Agent class ensures they stay in sync.
2. **Infer manifest from tools list**: Rejected because tool capabilities are abstract (what the agent needs conceptually) while tools are concrete (what's currently bound). An agent may need `data_query` capability but have no tool bound yet — that's the dormant state.
3. **Embed in system_prompt**: Rejected because manifests need to be machine-readable for registry/platform integration, not just LLM-readable.
4. **Use agent description field**: Rejected because description is free-text for humans. Manifests are structured data for machines.

## Implementation Plan

1. Add `src/strands/agent/manifest.py` with dataclass definitions
2. Add `manifest` parameter to `Agent.__init__` (optional, defaults to None)
3. Store manifest as `self.manifest` attribute on the Agent instance
4. Add serialization: `to_dict()`, `from_dict()`, `from_file()` (JSON and YAML support)
5. Export `AgentManifest`, `InputContract`, `OutputContract`, `Trigger` from `strands.agent`
6. Add unit tests for manifest creation, serialization, and Agent integration
7. Add documentation page explaining the concept and usage patterns
