Defined in: [src/memory/types.ts:171](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/types.ts#L171)

Options for [MemoryManager.search](/docs/api/typescript/MemoryManager/index.md#search).

Extends the store primitive [SearchOptions](/docs/api/typescript/SearchOptions/index.md) with manager-level store routing.

## Extends

-   [`SearchOptions`](/docs/api/typescript/SearchOptions/index.md)

## Properties

### maxSearchResults?

```ts
optional maxSearchResults?: number;
```

Defined in: [src/memory/types.ts:34](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/types.ts#L34)

Maximum number of results to return from this store.

#### Inherited from

[`SearchOptions`](/docs/api/typescript/SearchOptions/index.md).[`maxSearchResults`](/docs/api/typescript/SearchOptions/index.md#maxsearchresults)

---

### stores?

```ts
optional stores?: string[];
```

Defined in: [src/memory/types.ts:173](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/types.ts#L173)

Filter to specific stores by name. Omit to search all.