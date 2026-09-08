Defined in: [src/types/messages.ts:595](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L595)

Cache point block for prompt caching. Marks a position in a message or system prompt where caching should occur.

## Implements

-   [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md)
-   `JSONSerializable`<{ `cachePoint`: [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md); }>

## Constructors

### Constructor

```ts
new CachePointBlock(data): CachePointBlock;
```

Defined in: [src/types/messages.ts:612](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L612)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md) |

#### Returns

`CachePointBlock`

## Properties

### type

```ts
readonly type: "cachePointBlock";
```

Defined in: [src/types/messages.ts:599](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L599)

Discriminator for cache point.

---

### cacheType

```ts
readonly cacheType: "default";
```

Defined in: [src/types/messages.ts:604](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L604)

The cache type. Currently only ‘default’ is supported.

#### Implementation of

[`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md).[`cacheType`](/docs/api/typescript/CachePointBlockData/index.md#cachetype)

---

### ttl?

```ts
readonly optional ttl?: string;
```

Defined in: [src/types/messages.ts:610](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L610)

Optional TTL for the cache entry. See [CachePointBlockData.ttl](/docs/api/typescript/CachePointBlockData/index.md#ttl) for the provider-specific value space.

#### Implementation of

[`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md).[`ttl`](/docs/api/typescript/CachePointBlockData/index.md#ttl)

## Methods

### toJSON()

```ts
toJSON(): {
  cachePoint: CachePointBlockData;
};
```

Defined in: [src/types/messages.ts:623](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L623)

Serializes the CachePointBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify().

#### Returns

```ts
{
  cachePoint: CachePointBlockData;
}
```

| Name | Type | Defined in |
| --- | --- | --- |
| `cachePoint` | [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md) | [src/types/messages.ts:623](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L623) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): CachePointBlock;
```

Defined in: [src/types/messages.ts:638](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L638)

Creates a CachePointBlock instance from its wrapped data format.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `cachePoint`: [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md); } | Wrapped CachePointBlockData to deserialize |
| `data.cachePoint` | [`CachePointBlockData`](/docs/api/typescript/CachePointBlockData/index.md) | \- |

#### Returns

`CachePointBlock`

CachePointBlock instance