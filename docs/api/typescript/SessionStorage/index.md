```ts
type SessionStorage = {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/storage.ts:21](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/session/storage.ts#L21)

SessionStorage configuration for pluggable storage backends. Allows users to configure snapshot and transcript storage independently.

## Deprecated

Prefer passing a unified `Storage` directly to `SessionManagerConfig.storage`.

## Properties

### ~snapshot~

```ts
snapshot: SnapshotStorage;
```

Defined in: [src/session/storage.ts:22](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/session/storage.ts#L22)