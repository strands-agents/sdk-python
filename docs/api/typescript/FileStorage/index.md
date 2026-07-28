Defined in: [src/session/file-storage.ts:28](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L28)

File-based implementation of SnapshotStorage. Persists session snapshots to the local filesystem under a configurable base directory.

## Deprecated

Pass a unified `Storage` (e.g. `LocalFileStorage` from `@strands-agents/sdk/storage`) to `SessionManagerConfig.storage` instead. The session manager wraps it internally.

Directory layout:

```plaintext
<baseDir>/<sessionId>/scopes/<scope>/<scopeId>/snapshots/
  snapshot_latest.json
  immutable_history/
    snapshot_<uuid7>.json
```

## Implements

-   [`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md)

## Constructors

### Constructor

```ts
new FileStorage(baseDir): FileStorage;
```

Defined in: [src/session/file-storage.ts:35](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L35)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `baseDir` | `string` | Absolute path to the root directory for storing session snapshots. |

#### Returns

`FileStorage`

## Methods

### ~saveSnapshot()~

```ts
saveSnapshot(params): Promise<void>;
```

Defined in: [src/session/file-storage.ts:65](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L65)

Persists a snapshot to disk. If `isLatest` is true, writes to `snapshot_latest.json` (overwriting any previous). Otherwise, writes to `immutable_history/snapshot_<snapshotId>.json`.

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

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`saveSnapshot`](/docs/api/typescript/SnapshotStorage/index.md#savesnapshot)

---

### ~loadSnapshot()~

```ts
loadSnapshot(params): Promise<Snapshot>;
```

Defined in: [src/session/file-storage.ts:82](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L82)

Loads a snapshot from disk. If `snapshotId` is omitted, loads `snapshot_latest.json`. Returns null if the file does not exist.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `snapshotId?`: `string`; } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.snapshotId?` | `string` |

#### Returns

`Promise`<[`Snapshot`](/docs/api/typescript/Snapshot/index.md)\>

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`loadSnapshot`](/docs/api/typescript/SnapshotStorage/index.md#loadsnapshot)

---

### ~listSnapshotIds()~

```ts
listSnapshotIds(params): Promise<string[]>;
```

Defined in: [src/session/file-storage.ts:97](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L97)

Lists immutable snapshot IDs for a scope, sorted chronologically. Since IDs are UUID v7, lexicographic sort equals chronological order. `startAfter` filters to IDs after the given UUID v7 (exclusive cursor). `limit` caps the number of returned IDs. Returns an empty array if no snapshots exist yet.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `limit?`: `number`; `startAfter?`: `string`; } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.limit?` | `number` |
| `params.startAfter?` | `string` |

#### Returns

`Promise`<`string`\[\]>

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`listSnapshotIds`](/docs/api/typescript/SnapshotStorage/index.md#listsnapshotids)

---

### ~deleteSession()~

```ts
deleteSession(params): Promise<void>;
```

Defined in: [src/session/file-storage.ts:129](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L129)

Deletes all data for a session by removing its root directory (`<baseDir>/<sessionId>/`) recursively. No-ops if the session directory does not exist.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `sessionId`: `string`; } |
| `params.sessionId` | `string` |

#### Returns

`Promise`<`void`\>

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`deleteSession`](/docs/api/typescript/SnapshotStorage/index.md#deletesession)

---

### ~loadManifest()~

```ts
loadManifest(params): Promise<SnapshotManifest>;
```

Defined in: [src/session/file-storage.ts:143](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L143)

Loads the snapshot manifest for a scope. Returns a default manifest with the current timestamp if none exists yet.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |

#### Returns

`Promise`<[`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md)\>

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`loadManifest`](/docs/api/typescript/SnapshotStorage/index.md#loadmanifest)

---

### ~saveManifest()~

```ts
saveManifest(params): Promise<void>;
```

Defined in: [src/session/file-storage.ts:158](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/session/file-storage.ts#L158)

Persists the snapshot manifest for a scope to disk.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | { `location`: [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md); `manifest`: [`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md); } |
| `params.location` | [`SnapshotLocation`](/docs/api/typescript/SnapshotLocation/index.md) |
| `params.manifest` | [`SnapshotManifest`](/docs/api/typescript/SnapshotManifest/index.md) |

#### Returns

`Promise`<`void`\>

#### Implementation of

[`SnapshotStorage`](/docs/api/typescript/SnapshotStorage/index.md).[`saveManifest`](/docs/api/typescript/SnapshotStorage/index.md#savemanifest)