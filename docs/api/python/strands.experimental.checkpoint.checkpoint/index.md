Checkpoint system for durable agent execution.

A `Checkpoint` is a pause-point marker emitted at agent cycle boundaries. It captures the position (which boundary fired) and the cycle index. It does **not** capture conversation state — pair with a `SessionManager` for cross-process state continuity.

Positions per ReAct cycle:

-   `after_model`: model returned tool\_use; tools have not run yet.
-   `after_tools`: tools finished; the next model call has not happened yet.

Per-tool granularity within a cycle is the `ToolExecutor`’s responsibility.

Usage (mirrors interrupts):

-   Pause: `AgentResult` with `stop_reason="checkpoint"` and `checkpoint` populated.
-   Resume: pass back `\{"checkpointResume": \{"checkpoint": ckpt.to_dict()}}`.

Precedence:

-   Interrupt > checkpoint: an interrupt during a checkpointing cycle returns `stop_reason="interrupt"` and skips `after_tools`.
-   Cancel > checkpoint: a cancel signal at either boundary returns `stop_reason="cancelled"`.

**Notes**:

-   Checkpoints are only emitted on tool\_use cycles. A turn with no tool calls emits no checkpoint; use a `SessionManager` for durability of every turn.
-   `EventLoopMetrics` resets per invocation; aggregate yourself if needed.
-   `BeforeInvocationEvent` / `AfterInvocationEvent` fire on every resume, same as interrupts.

## Checkpoint

```python
@dataclass(frozen=True)
class Checkpoint()
```

Defined in: [src/strands/experimental/checkpoint/checkpoint.py:46](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/checkpoint/checkpoint.py#L46)

Pause-point marker. Treat as opaque — pass back to resume.

**Attributes**:

-   `position` - Which boundary fired (`after_model` or `after_tools`).
-   `cycle_index` - ReAct loop cycle (0-based).
-   `schema_version` - Rejects incompatible checkpoints on resume.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/experimental/checkpoint/checkpoint.py:59](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/checkpoint/checkpoint.py#L59)

Serialize for persistence.

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Checkpoint"
```

Defined in: [src/strands/experimental/checkpoint/checkpoint.py:64](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/checkpoint/checkpoint.py#L64)

Reconstruct from a dict produced by to\_dict().

**Arguments**:

-   `data` - Serialized checkpoint data.

**Raises**:

-   `CheckpointException` - If schema\_version doesn’t match the current version.