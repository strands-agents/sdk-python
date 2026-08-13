Defined in: [src/storage/storage.ts:74](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L74)

A backend for storing and retrieving raw bytes under string keys.

The interface is deliberately minimal — four operations over opaque `Uint8Array` values. Keys are opaque strings — implementations must round-trip the bytes they are given unchanged. The shipped backends interpret `/` as a logical separator (collapsing runs, rejecting `..`), but custom backends may apply their own key scheme.

The `ListQuery` type parameter controls what `list` accepts. It defaults to `string` (a key prefix), which every backend supports. Implementations may widen it to accept a richer query object (e.g. a DynamoDB partition/sort-key filter) while still accepting a plain string for SDK-internal callers.

Implement this to add a custom backend; the SDK ships InMemoryStorage, LocalFileStorage, and S3Storage.

## Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `ListQuery` | `string` |

## Methods

### write()

```ts
write(key, data): Promise<void>;
```

Defined in: [src/storage/storage.ts:82](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L82)

Stores `data` under `key`, overwriting any existing value.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `string` | Opaque string key identifying the value |
| `data` | `Uint8Array` | Raw bytes to persist |

#### Returns

`Promise`<`void`\>

#### Throws

[StorageError](/docs/api/typescript/StorageError/index.md) if the write fails

---

### read()

```ts
read(key): Promise<Uint8Array<ArrayBufferLike>>;
```

Defined in: [src/storage/storage.ts:91](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L91)

Retrieves the bytes previously stored under `key`.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `string` | The key to read |

#### Returns

`Promise`<`Uint8Array`<`ArrayBufferLike`\>>

The stored bytes, or `null` if no value exists for `key`

#### Throws

[StorageError](/docs/api/typescript/StorageError/index.md) if the read fails for a reason other than a missing key

---

### delete()

```ts
delete(key): Promise<void>;
```

Defined in: [src/storage/storage.ts:99](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L99)

Deletes the value stored under `key`. A no-op if the key does not exist.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `key` | `string` | The key to delete |

#### Returns

`Promise`<`void`\>

#### Throws

[StorageError](/docs/api/typescript/StorageError/index.md) if the delete fails

---

### list()

```ts
list(query): Promise<string[]>;
```

Defined in: [src/storage/storage.ts:115](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L115)

Lists keys matching the given query.

When `ListQuery` is `string` (the default), this is a prefix match — returns full keys (not the suffix after the prefix), sorted lexicographically. An empty string lists every key.

Implementations may accept richer query objects (e.g. partition + sort-key filters) while still supporting a plain string prefix for SDK-internal callers.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `ListQuery` | A string prefix or backend-specific query object |

#### Returns

`Promise`<`string`\[\]>

The matching keys, sorted ascending

#### Throws

[StorageError](/docs/api/typescript/StorageError/index.md) if the listing fails

---

### namespace()?

```ts
optional namespace(prefix): Storage;
```

Defined in: [src/storage/storage.ts:126](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/storage/storage.ts#L126)

Returns a view of this storage with all keys prefixed by `prefix`. The original storage is not mutated.

Optional — shipped backends implement this, custom backends may omit it.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `prefix` | `string` | Prefix to prepend to all keys |

#### Returns

`Storage`

A Storage view scoped to the given prefix