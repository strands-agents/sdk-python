Defined in: [src/session/session-manager.ts:59](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L59)

## Properties

### storage?

```ts
optional storage?:
  | Storage<string, string>
  | {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/session-manager.ts:70](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L70)

Storage backend for snapshot persistence.

Accepts either:

-   A unified [Storage](/docs/api/typescript/Storage/index.md) instance (recommended) — wrapped internally with SnapshotStorageAdapter
-   A legacy `{ snapshot: SnapshotStorage }` object

When omitted, resolves from the agent-level `storage` during initialization. If no agent-level storage is available either, an error is thrown.

---

### sessionId?

```ts
optional sessionId?: string;
```

Defined in: [src/session/session-manager.ts:72](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L72)

Unique session identifier. Defaults to `'default-session'`.

---

### saveLatestOn?

```ts
optional saveLatestOn?: SaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:74](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L74)

When to save snapshot\_latest. Default: `'invocation'` (after each agent invocation completes). See [SaveLatestStrategy](/docs/api/typescript/SaveLatestStrategy/index.md) for details.

---

### snapshotTrigger?

```ts
optional snapshotTrigger?: SnapshotTriggerCallback;
```

Defined in: [src/session/session-manager.ts:76](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L76)

Callback invoked after each invocation to decide whether to create an immutable snapshot.

---

### multiAgentSaveLatestOn?

```ts
optional multiAgentSaveLatestOn?: MultiAgentSaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:82](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/session/session-manager.ts#L82)

When to save snapshot\_latest for multi-agent orchestrators. Default: `'node'` (after each node invocation completes). See [MultiAgentSaveLatestStrategy](/docs/api/typescript/MultiAgentSaveLatestStrategy/index.md) for details.