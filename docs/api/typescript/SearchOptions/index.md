Defined in: [src/memory/types.ts:32](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/memory/types.ts#L32)

Options passed to [MemoryStore.search](/docs/api/typescript/MemoryStore/index.md#search).

Store implementations may extend this with backend-specific fields (e.g. a metadata filter or search-type override) in their own `search` signature. Note that [MemoryManager.search](/docs/api/typescript/MemoryManager/index.md#search) only forwards the base fields here across its (potentially heterogeneous) stores — to use a store’s extended options, call that store’s `search` directly, or set them as per-instance defaults on the store.

## Extended by

-   [`MemorySearchOptions`](/docs/api/typescript/MemorySearchOptions/index.md)

## Properties

### maxSearchResults?

```ts
optional maxSearchResults?: number;
```

Defined in: [src/memory/types.ts:34](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/memory/types.ts#L34)

Maximum number of results to return from this store.