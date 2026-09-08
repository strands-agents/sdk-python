Defined in: [src/session/session-manager.ts:62](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L62)

## Properties

### storage?

```ts
optional storage?:
  | Storage<string, string>
  | {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/session-manager.ts:73](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L73)

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

Defined in: [src/session/session-manager.ts:75](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L75)

Unique session identifier. Defaults to `'default-session'`.

---

### saveLatestOn?

```ts
optional saveLatestOn?: SaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:77](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L77)

When to save snapshot\_latest. Default: `'invocation'` (after each agent invocation completes). See [SaveLatestStrategy](/docs/api/typescript/SaveLatestStrategy/index.md) for details.

---

### snapshotTrigger?

```ts
optional snapshotTrigger?: SnapshotTriggerCallback;
```

Defined in: [src/session/session-manager.ts:79](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L79)

Callback invoked after each invocation to decide whether to create an immutable snapshot.

---

### multiAgentSaveLatestOn?

```ts
optional multiAgentSaveLatestOn?: MultiAgentSaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:85](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/session/session-manager.ts#L85)

When to save snapshot\_latest for multi-agent orchestrators. Default: `'node'` (after each node invocation completes). See [MultiAgentSaveLatestStrategy](/docs/api/typescript/MultiAgentSaveLatestStrategy/index.md) for details.