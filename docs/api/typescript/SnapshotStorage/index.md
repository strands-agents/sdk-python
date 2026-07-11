Defined in: [src/session/storage.ts:47](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L47)

Interface for snapshot persistence. Implementations provide storage backends (S3, filesystem, etc.).

File layout convention:

```plaintext
sessions/<session_id>/
  scopes/
    agent/<scope_id>/
      snapshots/
        snapshot_latest.json
        immutable_history/
          snapshot_<uuid>.json
          snapshot_<uuid>.json
```

## Methods

### saveSnapshot()

```ts
saveSnapshot(params): Promise<void>;
```

Defined in: [src/session/storage.ts:51](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L51)

Persists a snapshot to storage.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `snapshotId`: `string`; `isLatest`: `boolean`; `snapshot`: [`Snapshot`](/docs/api/typescript/Snapshot/index.md); } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.snapshotId` | `string` |
| `params.isLatest` | `boolean` |
| `params.snapshot` | [`Snapshot`](/docs/api/typescript/Snapshot/index.md) |

#### Returns

`Promise`<`void`\>

---

### loadSnapshot()

```ts
loadSnapshot(params): Promise<Snapshot>;
```

Defined in: [src/session/storage.ts:61](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L61)

Loads a snapshot from storage.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `snapshotId?`: `string`; } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.snapshotId?` | `string` |

#### Returns

`Promise`<[`Snapshot`](/docs/api/typescript/Snapshot/index.md)\>

---

### listSnapshotIds()

```ts
listSnapshotIds(params): Promise<string[]>;
```

Defined in: [src/session/storage.ts:77](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L77)

Lists all available immutable snapshot IDs for a session scope, sorted chronologically. Snapshot IDs are UUID v7 strings vended by the SDK — callers should treat them as opaque handles and never construct them manually.

Typical pagination pattern:

```typescript
const page1 = await storage.listSnapshotIds({ location })
const page2 = await storage.listSnapshotIds({ location, startAfter: page1.at(-1) })
```

`limit` caps the number of returned IDs. `startAfter` is an exclusive cursor (the last ID from the previous page); it must be a UUID v7 obtained from a prior `listSnapshotIds` call.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `limit?`: `number`; `startAfter?`: `string`; } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.limit?` | `number` |
| `params.startAfter?` | `string` |

#### Returns

`Promise`<`string`\[\]>

---

### deleteSession()

```ts
deleteSession(params): Promise<void>;
```

Defined in: [src/session/storage.ts:82](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L82)

Deletes all snapshots and directories belonging to the session ID.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `sessionId`: `string`; } |
| `params.sessionId` | `string` |

#### Returns

`Promise`<`void`\>

---

### loadManifest()

```ts
loadManifest(params): Promise<SnapshotManifest>;
```

Defined in: [src/session/storage.ts:87](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L87)

Loads the snapshot manifest.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |

#### Returns

`Promise`<[`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md)\>

---

### saveManifest()

```ts
saveManifest(params): Promise<void>;
```

Defined in: [src/session/storage.ts:92](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/session/storage.ts#L92)

Saves the snapshot manifest.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `manifest`: [`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md); } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.manifest` | [`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md) |

#### Returns

`Promise`<`void`\>