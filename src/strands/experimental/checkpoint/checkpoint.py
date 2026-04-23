"""Checkpoint system for durable agent execution.

Checkpoints enable crash-resilient agent workflows by capturing agent state at
cycle boundaries in the agent loop. A durability provider (e.g. Temporal) can
persist checkpoints and resume from them after failures.

Two checkpoint positions per ReAct cycle:
- after_model: model call completed, tools not yet executed.
- after_tools: all tools executed, next model call pending.

Per-tool granularity is handled by the ToolExecutor abstraction (e.g.
TemporalToolExecutor routes each tool to a separate Temporal activity).
The SDK checkpoint operates at cycle boundaries.

User-facing pattern (same as interrupts):
- Pause via stop_reason="checkpoint" on AgentResult
- State via AgentResult.checkpoint field
- Resume via checkpointResume content block in next agent() call

V0 Known Limitations:
- Metrics reset on each resume call. The caller is responsible for aggregating
  metrics across a durable run. EventLoopMetrics reflects only the current call.
- OpenAIResponsesModel(stateful=True) is not supported. The server-side
  response_id (_model_state) is not captured in the snapshot.
- When position is "after_tools", AgentResult.message is the assistant message
  that requested the tools; tool results are in the snapshot messages.
- BeforeInvocationEvent and AfterInvocationEvent fire on every resume call,
  same as interrupts. Hooks counting invocations will see each resume as a
  separate invocation.
- Per-tool granularity within a cycle requires a custom ToolExecutor
  (e.g. TemporalToolExecutor).
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "1.0"

CheckpointPosition = Literal["after_model", "after_tools"]


@dataclass
class Checkpoint:
    """Pause point in the agent loop. Treat as opaque — pass back to resume.

    Attributes:
        position: What just completed (after_model or after_tools).
        cycle_index: Which ReAct loop cycle (0-based).
        snapshot: Serialized agent state as a dict, produced by ``Snapshot.to_dict()``.
            Stored as ``dict[str, Any]`` (not a ``Snapshot`` object) because checkpoints
            must be JSON-serializable for cross-process persistence. The consumer
            reconstructs via ``Snapshot.from_dict()`` on resume.
        app_data: Application-level internal state data. The SDK does not read
            or modify this. Applications can store arbitrary data needed across
            checkpoint boundaries (e.g. session context, workflow metadata).
            Separate from ``Snapshot.app_data`` which captures agent-state-level
            data managed by the SDK.
        schema_version: Rejects mismatches on resume across schema versions.
    """

    position: CheckpointPosition
    cycle_index: int = 0
    snapshot: dict[str, Any] = field(default_factory=dict)
    app_data: dict[str, Any] = field(default_factory=dict)
    schema_version: str = field(init=False, default=CHECKPOINT_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Reconstruct from a dict produced by to_dict().

        Args:
            data: Serialized checkpoint data.

        Raises:
            ValueError: If schema_version doesn't match the current version.
        """
        version = data.get("schema_version", "")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoints with schema version {version!r} are not compatible "
                f"with current version {CHECKPOINT_SCHEMA_VERSION}."
            )
        known_keys = {k for k in cls.__dataclass_fields__ if k != "schema_version"}
        unknown_keys = set(data.keys()) - known_keys - {"schema_version"}
        if unknown_keys:
            logger.warning("unknown_keys=<%s> | ignoring unknown fields in checkpoint data", unknown_keys)
        return cls(**{k: v for k, v in data.items() if k in known_keys})
