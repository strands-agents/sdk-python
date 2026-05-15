"""Agent Manifest — Declarative Capability Profile.

This module provides the AgentManifest class and supporting dataclasses that allow agents to formally
declare their input requirements, output types, trigger conditions, and tool capabilities.

A manifest is purely declarative metadata. It does not change agent execution behavior. External systems
(registries, orchestrators, platforms) read manifests to manage agent lifecycle and composition.

Example:
    >>> from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger
    >>> manifest = AgentManifest(
    ...     name="data_analysis_agent",
    ...     version="1.0.0",
    ...     input_contract=InputContract(
    ...         features_required=["dataset_schema"],
    ...         events_consumed=["DataReadyEvent"],
    ...         tool_capabilities=["data_query"],
    ...     ),
    ...     output_contract=OutputContract(events_produced=["InsightEvent"]),
    ...     trigger=Trigger(type="event", condition="DataReadyEvent"),
    ... )
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class InputContract:
    """Declares what an agent requires to execute.

    Attributes:
        features_required: Named features the agent needs from a feature store or data source.
        data_sources: Named data sources the agent reads from (databases, APIs, streams).
        events_consumed: Event types that this agent processes as input.
        tool_capabilities: Abstract tool capability names the agent requires (not concrete tool implementations).
        knowledge_bases: Named knowledge bases the agent queries during reasoning.
    """

    features_required: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    events_consumed: list[str] = field(default_factory=list)
    tool_capabilities: list[str] = field(default_factory=list)
    knowledge_bases: list[str] = field(default_factory=list)


@dataclass
class OutputContract:
    """Declares what an agent produces.

    Attributes:
        events_produced: Event types that this agent emits as output.
        features_produced: Named features that this agent computes and writes.
        artifacts_produced: Named artifacts (files, reports, records) that this agent creates.
    """

    events_produced: list[str] = field(default_factory=list)
    features_produced: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)


@dataclass
class Trigger:
    """Declares what activates this agent.

    Attributes:
        type: The trigger mechanism. One of "event", "schedule", or "on_demand".
        condition: For "event" type: the event type filter. For "schedule" type: a cron expression.
            For "on_demand" type: None (activated by explicit invocation).
    """

    type: str  # "event", "schedule", "on_demand"
    condition: Optional[str] = None


@dataclass
class AgentManifest:
    """Declarative capability profile for an agent.

    A manifest describes what an agent needs (inputs), what it produces (outputs),
    what activates it (trigger), and what scenarios it supports. It is purely
    declarative metadata that does not change agent execution behavior.

    External systems (registries, orchestrators, platforms) read manifests to:
    - Determine if an agent's dependencies are satisfiable
    - Manage agent lifecycle (dormant/active states)
    - Compose agents into chains by matching outputs to inputs
    - Select agents for scenario execution based on capabilities

    Attributes:
        name: Unique identifier for this agent.
        version: Semantic version string (e.g., "1.0.0").
        domain: Optional domain or category this agent belongs to.
        description: Human-readable description of what this agent does.
        input_contract: What this agent requires to execute.
        output_contract: What this agent produces.
        trigger: What activates this agent.
        scenario_types: List of scenario type identifiers this agent supports.
        tags: Arbitrary key-value metadata for filtering and discovery.

    Example:
        >>> manifest = AgentManifest(
        ...     name="enrichment_agent",
        ...     version="1.0.0",
        ...     domain="data_pipeline",
        ...     input_contract=InputContract(
        ...         events_consumed=["RawDataEvent"],
        ...         tool_capabilities=["data_query", "enrichment_api"],
        ...     ),
        ...     output_contract=OutputContract(events_produced=["EnrichedDataEvent"]),
        ...     trigger=Trigger(type="event", condition="RawDataEvent"),
        ... )
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a dictionary.

        Returns:
            Dictionary representation of the manifest, suitable for JSON serialization
            or registry registration.
        """
        data = asdict(self)
        # Remove None values for cleaner serialization
        if data.get("trigger") is None:
            del data["trigger"]
        if data.get("domain") is None:
            del data["domain"]
        if data.get("description") is None:
            del data["description"]
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation. Defaults to 2.

        Returns:
            JSON string representation of the manifest.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentManifest":
        """Deserialize manifest from a dictionary.

        Args:
            data: Dictionary containing manifest fields.

        Returns:
            AgentManifest instance.
        """
        input_contract = InputContract(**data.get("input_contract", {}))
        output_contract = OutputContract(**data.get("output_contract", {}))
        trigger = Trigger(**data["trigger"]) if data.get("trigger") else None

        return cls(
            name=data["name"],
            version=data["version"],
            domain=data.get("domain"),
            description=data.get("description"),
            input_contract=input_contract,
            output_contract=output_contract,
            trigger=trigger,
            scenario_types=data.get("scenario_types", []),
            tags=data.get("tags", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentManifest":
        """Load manifest from a JSON file.

        Args:
            path: Path to the manifest JSON file.

        Returns:
            AgentManifest instance.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def satisfies(self, available_events: list[str] | None = None,
                  available_features: list[str] | None = None,
                  available_capabilities: list[str] | None = None) -> tuple[bool, list[str]]:
        """Check if this manifest's input requirements can be satisfied by available resources.

        Args:
            available_events: List of event types currently available in the system.
            available_features: List of feature names currently available.
            available_capabilities: List of tool capability names currently registered.

        Returns:
            Tuple of (is_satisfied, unmet_requirements). If is_satisfied is True,
            all input requirements are met. If False, unmet_requirements lists what's missing.
        """
        unmet: list[str] = []

        if available_events is not None:
            for event in self.input_contract.events_consumed:
                if event not in available_events:
                    unmet.append(f"event:{event}")

        if available_features is not None:
            for feature in self.input_contract.features_required:
                if feature not in available_features:
                    unmet.append(f"feature:{feature}")

        if available_capabilities is not None:
            for cap in self.input_contract.tool_capabilities:
                if cap not in available_capabilities:
                    unmet.append(f"capability:{cap}")

        return (len(unmet) == 0, unmet)


# ─────────────────────────────────────────────────────────────────────────────
# Usage Examples
# ─────────────────────────────────────────────────────────────────────────────
#
# Example 1: Data pipeline agent that enriches raw events
#
#   from strands import Agent, tool
#   from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger
#
#   @tool
#   def enrich_record(record_id: str) -> dict:
#       """Look up additional context for a record."""
#       ...
#
#   enrichment_agent = Agent(
#       manifest=AgentManifest(
#           name="enrichment_agent",
#           version="1.0.0",
#           domain="data_pipeline",
#           description="Enriches raw ingestion events with contextual metadata",
#           input_contract=InputContract(
#               events_consumed=["RawIngestionEvent"],
#               features_required=["entity_profile"],
#               tool_capabilities=["lookup_api"],
#           ),
#           output_contract=OutputContract(
#               events_produced=["EnrichedEvent"],
#           ),
#           trigger=Trigger(type="event", condition="RawIngestionEvent"),
#       ),
#       system_prompt="You enrich raw data records with contextual metadata...",
#       tools=[enrich_record],
#   )
#
#
# Example 2: Checking if an agent's dependencies are satisfiable
#
#   manifest = enrichment_agent.manifest
#   satisfied, gaps = manifest.satisfies(
#       available_events=["RawIngestionEvent"],
#       available_features=[],  # entity_profile not yet available
#       available_capabilities=["lookup_api"],
#   )
#   # satisfied = False
#   # gaps = ["feature:entity_profile"]
#
#
# Example 3: Multi-agent composition — platform reads manifests to auto-wire
#
#   # Agent A produces "AnalysisCompleteEvent"
#   # Agent B consumes "AnalysisCompleteEvent"
#   # A platform reads both manifests and connects them without custom code:
#   #
#   #   for agent in registered_agents:
#   #       for event in agent.manifest.output_contract.events_produced:
#   #           downstream = [a for a in registered_agents
#   #                         if event in a.manifest.input_contract.events_consumed]
#   #           register_event_route(event, downstream)
#
#
# Example 4: Loading manifest from a file (useful for CI/CD validation)
#
#   # manifest.json:
#   # {
#   #   "name": "report_generator",
#   #   "version": "2.1.0",
#   #   "input_contract": {
#   #     "features_required": ["monthly_metrics", "comparison_baseline"],
#   #     "tool_capabilities": ["chart_generation", "pdf_export"]
#   #   },
#   #   "output_contract": {
#   #     "artifacts_produced": ["monthly_report.pdf"]
#   #   },
#   #   "trigger": {"type": "schedule", "condition": "0 9 1 * *"}
#   # }
#   #
#   # agent = Agent(
#   #     manifest=AgentManifest.from_file("manifest.json"),
#   #     system_prompt="...",
#   #     tools=[...],
#   # )

