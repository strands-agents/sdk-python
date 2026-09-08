```ts
type SessionStorage = {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/storage.ts:21](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/session/storage.ts#L21)

SessionStorage configuration for pluggable storage backends. Allows users to configure snapshot and transcript storage independently.

## Deprecated

Prefer passing a unified `Storage` directly to `SessionManagerConfig.storage`.

## Properties

### ~snapshot~

```ts
snapshot: SnapshotStorage;
```

Defined in: [src/session/storage.ts:22](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/session/storage.ts#L22)