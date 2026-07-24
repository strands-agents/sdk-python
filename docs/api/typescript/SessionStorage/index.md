```ts
type SessionStorage = {
  snapshot: SnapshotStorage;
};
```

Defined in: [src/session/storage.ts:21](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/storage.ts#L21)

SessionStorage configuration for pluggable storage backends. Allows users to configure snapshot and transcript storage independently.

## Deprecated

Prefer passing a unified `Storage` directly to `SessionManagerConfig.storage`.

## Properties

### ~snapshot~

```ts
snapshot: SnapshotStorage;
```

Defined in: [src/session/storage.ts:22](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/session/storage.ts#L22)