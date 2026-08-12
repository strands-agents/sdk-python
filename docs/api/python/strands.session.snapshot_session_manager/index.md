Snapshot-based session manager.

Persists an agent as a single versioned :class:`~strands.types._snapshot.Snapshot` blob on each lifecycle event, mirroring the TypeScript SDK’s `SessionManager`:

-   A mutable `snapshot_latest` is overwritten on each save, for crash/restart resume.
-   Append-only immutable snapshots (time-ordered keys) are written when a `snapshot_trigger` fires, enabling checkpointing — restore to any prior state, not just the latest.

The manager persists snapshots through the unified :class:`~strands.storage.storage.Storage` primitive (`write`/`read`/`delete`/`list` over byte blobs). It owns the key layout, snapshot-id scheme, and serialization; the storage backend only moves bytes, so the same `Storage` instance can back sessions, memory, and other subsystems.

This is distinct from the older message-log session managers (:class:`~strands.session.repository_session_manager.RepositorySessionManager` and its subclasses), which persist each message individually. Snapshots capture the whole agent in one atomic blob and are the recommended path for new agents.

#### SaveLatestStrategy

Controls how often `snapshot_latest` is saved automatically.

-   `"invocation"`: after every agent invocation completes (default; balances durability and I/O).
-   `"message"`: after every message added (most durable, highest I/O).
-   `"trigger"`: only when `snapshot_trigger` fires (or manually via `save_snapshot`).

Guardrail redactions are flushed immediately under every strategy, including `"trigger"`, so pre-redaction content never sits at rest. This diverges from the TypeScript SDK, which does not flush redactions under `"trigger"`; see :meth:`SnapshotSessionManager.redact_latest_message`.

## SnapshotTrigger

```python
@runtime_checkable
class SnapshotTrigger(Protocol)
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:167](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L167)

Decides whether to write an immutable checkpoint after an invocation.

#### \_\_call\_\_

```python
def __call__(*, agent_data: "Agent", **kwargs: Any) -> bool
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:170](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L170)

Return True to append an immutable snapshot for the given agent.

**Arguments**:

-   `agent_data` - The agent that just completed an invocation.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Returns**:

True to create an immutable checkpoint, False otherwise.

## SnapshotSessionManager

```python
class SnapshotSessionManager(SessionManager)
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:183](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L183)

Persists agent snapshots to a :class:`~strands.storage.storage.Storage` across invocations.

On agent initialization the latest snapshot is restored automatically. On each qualifying lifecycle event the agent is re-captured and `snapshot_latest` is overwritten. When `snapshot_trigger` returns True after an invocation, an additional immutable snapshot is appended for time-travel restore.

Single agents only. Attaching this manager to a Graph or Swarm raises `NotImplementedError`; use a message-log session manager for orchestrators.

**Example**:

```python
from strands import Agent
from strands.session import SnapshotSessionManager
from strands.storage import LocalFileStorage

session = SnapshotSessionManager("my-session", storage=LocalFileStorage())
agent = Agent(session_manager=session)
```

#### \_\_init\_\_

```python
def __init__(session_id: str = "default-session",
             *,
             storage: Storage | None = None,
             save_latest_on: SaveLatestStrategy = "invocation",
             snapshot_trigger: SnapshotTrigger | None = None,
             **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:205](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L205)

Initialize the snapshot session manager.

**Arguments**:

-   `session_id` - Unique session identifier. Must not contain path separators.
-   `storage` - Unified storage backend that persists snapshot blobs. When None, resolves from the agent-level `storage` during initialization; if no agent-level storage is available, falls back to :class:`~strands.storage.local_file_storage.LocalFileStorage`.
-   `save_latest_on` - When to overwrite `snapshot_latest`. See :data:`SaveLatestStrategy`.
-   `snapshot_trigger` - Optional callback invoked after each invocation; when it returns True an immutable snapshot is appended for checkpointing. An immutable snapshot can also be forced at any point via :meth:`save_snapshot`.
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Raises**:

-   `ValueError` - If `session_id` is empty, is a relative-path segment (`.` or `..`), normalizes to empty, or contains a path separator; or if `save_latest_on` is not a recognized strategy.

#### register\_hooks

```python
def register_hooks(registry: HookRegistry, **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:257](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L257)

Register lifecycle callbacks for snapshot persistence.

Overrides the base wiring: the message-log callbacks are replaced with snapshot save/restore handlers.

#### initialize

```python
def initialize(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:296](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L296)

Restore the agent from its latest snapshot, if one exists.

Storage is resolved on the first call and cached; a single manager instance should not be shared across agents with differing storage backends.

**Arguments**:

-   `agent` - Agent to restore.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### sync\_agent

```python
def sync_agent(agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:310](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L310)

Capture the agent and overwrite `snapshot_latest`.

**Arguments**:

-   `agent` - Agent to persist.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### redact\_latest\_message

```python
def redact_latest_message(redact_message: Message, agent: "Agent",
                          **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:319](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L319)

Persist immediately after a guardrail redaction, under every strategy.

The Agent has already applied the redaction to `agent.messages[-1]` before calling this, so re-capturing the agent flushes pre-redaction content out of the persisted latest snapshot. This flush happens regardless of `save_latest_on` (including `"trigger"`) because the Agent invokes this method directly, not through a hook the manager could decline to register — so pre-redaction content never sits at rest. This diverges from the TypeScript SDK, which gates redaction persistence behind an `AfterModelCall` hook it skips under `"trigger"` and therefore does not flush there.

**Arguments**:

-   `redact_message` - The redacted replacement message (already applied by the Agent).
-   `agent` - Agent whose latest message was redacted.
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### append\_message

```python
def append_message(message: Message, agent: "Agent", **kwargs: Any) -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:337](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L337)

No-op — snapshots capture the whole agent.

Per-message persistence under the `"message"` strategy is handled by the `MessageAddedEvent` hook, not by this method.

**Arguments**:

-   `message` - The message that was appended (unused).
-   `agent` - The agent the message was appended to (unused).
-   `**kwargs` - Additional keyword arguments for future extensibility.

#### list\_snapshot\_ids

```python
async def list_snapshot_ids(agent: "Agent",
                            *,
                            limit: int | None = None,
                            start_after: str | None = None) -> list[str]
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:351](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L351)

List immutable snapshot ids for an agent, oldest first.

**Arguments**:

-   `agent` - Agent whose snapshots to list.
-   `limit` - Optional cap on the number of ids returned.
-   `start_after` - Exclusive cursor; a snapshot id from a prior page.

**Returns**:

Immutable snapshot ids in chronological order.

**Raises**:

-   `ValueError` - If `start_after` is not a valid snapshot id.

#### restore\_snapshot

```python
async def restore_snapshot(agent: "Agent",
                           *,
                           snapshot_id: str | None = None) -> bool
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:381](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L381)

Restore an agent from a stored snapshot.

**Arguments**:

-   `agent` - Agent to restore into.
-   `snapshot_id` - The immutable snapshot id to restore (time travel). Omit to restore `snapshot_latest`, the same snapshot restore-on-init loads.

**Returns**:

True if the snapshot existed and was restored, False otherwise.

**Raises**:

-   `ValueError` - If `snapshot_id` is given and is not a valid snapshot id.

#### save\_snapshot

```python
async def save_snapshot(agent: "Agent", *, is_latest: bool) -> str | None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:397](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L397)

Save a snapshot of the agent’s current state on demand.

Use `is_latest=False` to force an immutable checkpoint at an arbitrary point (independent of `snapshot_trigger`), so it can later be restored with :meth:`restore_snapshot`; use `is_latest=True` to overwrite `snapshot_latest`.

**Arguments**:

-   `agent` - Agent whose state to capture.
-   `is_latest` - When True, overwrite `snapshot_latest` (a single mutable snapshot). When False, append a new immutable snapshot under a fresh, time-ordered id.

**Returns**:

The new immutable snapshot id, ready to pass to :meth:`restore_snapshot`, or `None` when `is_latest=True` (`snapshot_latest` is not addressed by id).

#### delete\_session

```python
async def delete_session() -> None
```

Defined in: [src/strands/session/snapshot\_session\_manager.py:419](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/session/snapshot_session_manager.py#L419)

Delete all snapshots for this session.