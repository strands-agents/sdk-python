"""Checkpoint types for durable agent execution.

⚠️ Experimental — APIs may change in future releases.

Known limitations (V1):

1. Hooks: BeforeInvocationEvent/AfterInvocationEvent, BeforeModelCallEvent/AfterModelCallEvent,
   and BeforeToolCallEvent/AfterToolCallEvent are not fired. No tracing spans.
2. Model retry: ModelRetryStrategy is bypassed; the orchestrator handles retries.
3. Structured output: structured_output_model / structured_output_prompt not supported.
4. Conversation management: apply_management / reduce_context do not run between
   checkpoint steps. The caller manages conversation after the final result.
5. Streaming: Token-by-token events are not re-emitted on replay.
   Use callback_handler=None for durable agents.
6. MCP: Remote servers (HTTP/SSE) work. Stdio servers do not survive worker crashes.
7. invocation_state is not supported in checkpoint mode.
8. Stateful models: OpenAIResponsesModel(stateful=True) is not supported.
   The server-side response_id is not preserved across checkpoints.
   Use stateful=False or OpenAIModel (Chat Completions) for durable execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ...types.streaming import StopReason
from ...types.tools import ToolResult

CHECKPOINT_SCHEMA_VERSION = "1.0"

CheckpointPosition = Literal["after_model", "after_tool", "after_tools"]


@dataclass
class Checkpoint:
    """Pause point in the agent loop. Treat as opaque — pass back to resume.

    Three positions:
    - after_model: model call completed, no tools executed yet.
    - after_tool: one tool executed, more tools remaining.
    - after_tools: all tools executed, next model call not yet started.

    Attributes:
        schema_version: Version of the checkpoint schema. Used to detect
            incompatible checkpoints across SDK upgrades. The SDK will reject
            checkpoints with a different schema_version.
        position: Where in the loop this checkpoint was taken.
        stop_reason: Why the model stopped (e.g. "tool_use", "end_turn").
        cycle_index: Which model→tools cycle this checkpoint is in (0-based).
        tool_index: Index of the next tool to execute (0-based). Only meaningful
            at after_model and after_tool positions.
        completed_tool_results: Tool results accumulated so far in this cycle.
            Carried across after_tool checkpoints. Appended to messages when
            the last tool finishes.
        snapshot: Agent state snapshot for restoration. Contains messages,
            state, conversation_manager_state, and interrupt_state.
        app_data: Provider-owned arbitrary data. The SDK does not read or modify this.
            Durability providers can use this to store provider-specific metadata
            (e.g. Temporal workflow IDs, Lambda step tokens, retry counts).
    """

    position: CheckpointPosition
    stop_reason: StopReason
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    cycle_index: int = 0
    tool_index: int = 0
    completed_tool_results: list[ToolResult] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    app_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence (e.g. Temporal Event History)."""
        return {
            "schema_version": self.schema_version,
            "position": self.position,
            "stop_reason": self.stop_reason,
            "cycle_index": self.cycle_index,
            "tool_index": self.tool_index,
            "completed_tool_results": self.completed_tool_results,
            "snapshot": self.snapshot,
            "app_data": self.app_data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        """Reconstruct from a dict produced by to_dict().

        Raises:
            ValueError: If schema_version doesn't match the current version.
        """
        version = d.get("schema_version", "")
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible checkpoint schema version: {version!r}. "
                f"Current version: {CHECKPOINT_SCHEMA_VERSION}. "
                f"Checkpoints from a different SDK version cannot be resumed."
            )
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
