Defined in: [src/memory/extraction/types.ts:130](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L130)

Per-store automatic-extraction configuration.

Lives on a store (via [MemoryStoreConfig](/docs/api/typescript/MemoryStoreConfig/index.md)) so different stores can extract on different schedules and in different styles. [trigger](#trigger) decides *when*; [extractor](#extractor) decides *how* (omit it to pass raw messages straight to the store); [filter](#filter) prunes content blocks first.

## Properties

### trigger?

```ts
optional trigger?:
  | ExtractionTrigger
  | ExtractionTrigger[];
```

Defined in: [src/memory/extraction/types.ts:136](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L136)

When to run extraction. A single trigger or an array; multiple triggers compose (extraction runs whenever any of them fires). Omit to default to every 5 turns; an explicit empty array is rejected at construction.

---

### extractor?

```ts
optional extractor?: Extractor;
```

Defined in: [src/memory/extraction/types.ts:144](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L144)

How to turn messages into entries (client-side extraction). When set, the store must implement `add` and each produced entry is stored through it. When omitted, the default depends on the store’s write methods: a store implementing `addMessages` uses server-side extraction (the manager hands it the raw messages, no model call), while a store implementing only `add` defaults to a [ModelExtractor](/docs/api/typescript/ModelExtractor/index.md) that distills facts client-side.

---

### filter?

```ts
optional filter?: MemoryMessageFilter;
```

Defined in: [src/memory/extraction/types.ts:152](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/memory/extraction/types.ts#L152)

Content blocks to strip before extraction. Defaults to DEFAULT\_MEMORY\_MESSAGE\_FILTER (excludes `toolUse` / `toolResult`).

For use cases or extractors with value in distilling over the *full* turn you instead want everything: pass `{ exclude: [] }` so tool blocks reach `addMessages`.