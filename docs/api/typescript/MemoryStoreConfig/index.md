Defined in: [src/memory/types.ts:61](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L61)

Declarative properties shared by every memory store and its config.

This is the single source of truth for a store’s identity and behavior knobs. Both the runtime [MemoryStore](/docs/api/typescript/MemoryStore/index.md) interface and concrete store configs extend it, so these fields are declared once. Concrete stores add their own backend-specific config fields on top.

## Extended by

-   [`MemoryStore`](/docs/api/typescript/MemoryStore/index.md)

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/memory/types.ts:63](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L63)

Identifier for this store, used to target specific stores in search/add tools. Must be unique.

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/memory/types.ts:65](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L65)

Human-readable description of what this store contains. Included in tool descriptions.

---

### maxSearchResults?

```ts
readonly optional maxSearchResults?: number;
```

Defined in: [src/memory/types.ts:70](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L70)

Default maximum number of results this store returns per search, used when a caller does not pass a per-call `maxSearchResults`.

---

### writable?

```ts
readonly optional writable?: boolean;
```

Defined in: [src/memory/types.ts:77](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L77)

Whether this store accepts writes. Optional at config time (caller intent, defaults to `false`); concrete stores resolve it to a definite boolean on the [MemoryStore](/docs/api/typescript/MemoryStore/index.md) interface.

#### Default Value

```ts
false
```

---

### extraction?

```ts
readonly optional extraction?: boolean | ExtractionConfig;
```

Defined in: [src/memory/types.ts:90](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/memory/types.ts#L90)

Automatic-extraction config for this writable store, as a `boolean | config` shorthand. `true` enables it with defaults; an [ExtractionConfig](/docs/api/typescript/ExtractionConfig/index.md) defaults any unset field; `false`/omitted is off.

The defaults run every 5 turns, and the extraction method depends on the store’s write methods. A store implementing `addMessages` uses server-side extraction: the manager hands it the raw messages and the backend extracts them, with no model call. A store implementing only `add` uses a [ModelExtractor](/docs/api/typescript/ModelExtractor/index.md) for client-side extraction: it calls the agent’s model to distill facts and stores each one via `add`.

#### Default Value

```ts
false
```