Defined in: [src/errors.ts:264](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/errors.ts#L264)

Error thrown when a storage operation fails.

Wraps backend-specific errors (filesystem, S3, network) with a uniform type that consumers can catch without knowing which backend is in use.

## Extends

-   `Error`

## Constructors

### Constructor

```ts
new StorageError(message, options?): StorageError;
```

Defined in: [src/errors.ts:271](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/errors.ts#L271)

Creates a new StorageError.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `message` | `string` | Error message describing the storage failure |
| `options?` | `ErrorOptions` | Optional error options including cause for error chaining |

#### Returns

`StorageError`

#### Overrides

```ts
Error.constructor
```