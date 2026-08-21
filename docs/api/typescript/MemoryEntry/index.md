Defined in: [src/memory/types.ts:10](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L10)

A single memory entry retrieved from or stored to a memory store.

## Properties

### content

```ts
content: string;
```

Defined in: [src/memory/types.ts:12](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L12)

The textual content of this memory entry.

---

### storeName?

```ts
optional storeName?: string;
```

Defined in: [src/memory/types.ts:18](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L18)

Name of the store this entry came from. Populated by [MemoryManager.search](/docs/api/typescript/MemoryManager/index.md#search) so callers (and the model, via `search_memory`) can tell which store produced each result and refine targeting. Stores need not set this themselves.

---

### metadata?

```ts
optional metadata?: Record<string, JSONValue>;
```

Defined in: [src/memory/types.ts:20](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L20)

Optional metadata (e.g., score, source, id, timestamp).