Defined in: [src/session/session-manager.ts:59](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L59)

## Properties

### storage

```ts
storage:
  | Storage<string>
  | {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/session-manager.ts:67](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L67)

Storage backend for snapshot persistence.

Accepts either:

-   A unified [Storage](/docs/api/typescript/Storage/index.md) instance (recommended) — wrapped internally with SnapshotStorageAdapter
-   A legacy `{ snapshot: SnapshotStorage }` object

---

### sessionId?

```ts
optional sessionId?: string;
```

Defined in: [src/session/session-manager.ts:69](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L69)

Unique session identifier. Defaults to `'default-session'`.

---

### saveLatestOn?

```ts
optional saveLatestOn?: SaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:71](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L71)

When to save snapshot\_latest. Default: `'invocation'` (after each agent invocation completes). See [SaveLatestStrategy](/docs/api/typescript/SaveLatestStrategy/index.md) for details.

---

### snapshotTrigger?

```ts
optional snapshotTrigger?: SnapshotTriggerCallback;
```

Defined in: [src/session/session-manager.ts:73](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L73)

Callback invoked after each invocation to decide whether to create an immutable snapshot.

---

### multiAgentSaveLatestOn?

```ts
optional multiAgentSaveLatestOn?: MultiAgentSaveLatestStrategy;
```

Defined in: [src/session/session-manager.ts:79](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/session-manager.ts#L79)

When to save snapshot\_latest for multi-agent orchestrators. Default: `'node'` (after each node invocation completes). See [MultiAgentSaveLatestStrategy](/docs/api/typescript/MultiAgentSaveLatestStrategy/index.md) for details.