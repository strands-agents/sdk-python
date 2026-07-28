```ts
type SaveLatestStrategy = "message" | "invocation" | "trigger";
```

Defined in: [src/session/session-manager.ts:47](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/session-manager.ts#L47)

Controls when `snapshot_latest` is saved automatically for agents.

There are two kinds of snapshots:

-   **`snapshot_latest`**: A single mutable snapshot that is overwritten on each save. Used to resume the most recent conversation state (e.g. after a crash or restart). Always reflects the last saved point in time.
-   **Immutable snapshots**: Append-only snapshots with unique IDs (UUID v7), created only when `snapshotTrigger` fires. Used for checkpointing — you can restore to any prior state, not just the latest.

`SaveLatestStrategy` controls how frequently `snapshot_latest` is updated:

-   `'invocation'`: after every agent invocation completes (default; balances durability and I/O)
-   `'message'`: after every message added and when an invocation completes (most durable, highest I/O)
-   `'trigger'`: only when a `snapshotTrigger` fires (or manually via `saveSnapshot`)

Under `'invocation'` and `'message'`, guardrail redactions are persisted immediately so pre-redaction content never sits at rest. Under `'trigger'`, the caller’s `snapshotTrigger` stays in control; redactions are only flushed if the trigger fires or `saveSnapshot` is called.