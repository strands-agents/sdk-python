"""Checkpoint system for durable agent execution.

Checkpoints capture agent state at cycle boundaries so a durability provider
(e.g. Temporal) can persist them and resume after failures.

Positions per ReAct cycle:
- ``after_model``: model call completed, tools not yet executed.
- ``after_tools``: all tools executed, next model call pending.

Per-tool granularity within a cycle is the ToolExecutor's responsibility
(e.g. TemporalToolExecutor routes each tool to a separate activity).

Usage (mirrors the interrupt pattern):
- Pause: ``AgentResult`` with ``stop_reason="checkpoint"`` and a populated
  ``checkpoint`` field.
- Resume: pass ``[{"checkpointResume": {"checkpoint": checkpoint.to_dict()}}]``
  as the next prompt.

Precedence:
- Interrupts > checkpoint: an interrupt raised during a checkpointing cycle
  returns ``stop_reason="interrupt"`` and skips the ``after_tools`` checkpoint.
- Cancel > checkpoint: a cancel signal set at either checkpoint boundary
  suppresses emission and surfaces as ``stop_reason="cancelled"``.

Known limitations:
- ``EventLoopMetrics`` resets per invocation; aggregate across resumes yourself.
- ``OpenAIResponsesModel(stateful=True)`` is not supported — the server-side
  ``response_id`` is not captured.
- At ``after_tools``, ``AgentResult.message`` is the assistant's tool-use
  message; tool results live in the snapshot.
- ``BeforeInvocationEvent`` / ``AfterInvocationEvent`` fire on every resume,
  same as interrupts.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ...types.exceptions import CheckpointException

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = "1.0"

CheckpointPosition = Literal["after_model", "after_tools"]


@dataclass(frozen=True)
class Checkpoint:
    """Pause point in the agent loop. Treat as opaque — pass back to resume.

    Attributes:
        position: What just completed (``after_model`` or ``after_tools``).
        cycle_index: ReAct loop cycle (0-based).
        snapshot: Serialized agent state from ``Snapshot.to_dict()``. Stored as
            a dict (not a ``Snapshot``) so the checkpoint is JSON-serializable.
        app_data: Opaque application-level state. The SDK does not read or
            modify this. Distinct from ``Snapshot.app_data`` (agent-level).
        schema_version: Used to reject incompatible checkpoints on resume.
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
            CheckpointException: If schema_version doesn't match the current version.
        """
        version = data.get("schema_version", "")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointException(
                f"Checkpoints with schema version {version!r} are not compatible "
                f"with current version {CHECKPOINT_SCHEMA_VERSION}."
            )
        known_keys = {k for k in cls.__dataclass_fields__ if k != "schema_version"}
        unknown_keys = set(data.keys()) - known_keys - {"schema_version"}
        if unknown_keys:
            logger.warning("unknown_keys=<%s> | ignoring unknown fields in checkpoint data", unknown_keys)
        return cls(**{k: v for k, v in data.items() if k in known_keys})
