Defined in: [src/memory/types.ts:101](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L101)

Interface for a memory store backend.

Extends [MemoryStoreConfig](/docs/api/typescript/MemoryStoreConfig/index.md) with the runtime methods a store provides. Every store is searchable; the resolved `writable` flag declares whether it also accepts writes, which is how the [MemoryManager](/docs/api/typescript/MemoryManager/index.md) decides where to route them. `search_memory` can query all stores, while `add_memory` can only write to `writable` stores.

## Extends

-   [`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md)

## Properties

### name

```ts
readonly name: string;
```

Defined in: [src/memory/types.ts:63](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L63)

Identifier for this store, used to target specific stores in search/add tools. Must be unique.

#### Inherited from

[`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md).[`name`](/docs/api/typescript/MemoryStoreConfig/index.md#name)

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/memory/types.ts:65](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L65)

Human-readable description of what this store contains. Included in tool descriptions.

#### Inherited from

[`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md).[`description`](/docs/api/typescript/MemoryStoreConfig/index.md#description)

---

### maxSearchResults?

```ts
readonly optional maxSearchResults?: number;
```

Defined in: [src/memory/types.ts:70](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L70)

Default maximum number of results this store returns per search, used when a caller does not pass a per-call `maxSearchResults`.

#### Inherited from

[`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md).[`maxSearchResults`](/docs/api/typescript/MemoryStoreConfig/index.md#maxsearchresults)

---

### extraction?

```ts
readonly optional extraction?: boolean | ExtractionConfig;
```

Defined in: [src/memory/types.ts:90](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L90)

Automatic-extraction config for this writable store, as a `boolean | config` shorthand. `true` enables it with defaults; an [ExtractionConfig](/docs/api/typescript/ExtractionConfig/index.md) defaults any unset field; `false`/omitted is off.

The defaults run every 5 turns, and the extraction method depends on the store’s write methods. A store implementing `addMessages` uses server-side extraction: the manager hands it the raw messages and the backend extracts them, with no model call. A store implementing only `add` uses a [ModelExtractor](/docs/api/typescript/ModelExtractor/index.md) for client-side extraction: it calls the agent’s model to distill facts and stores each one via `add`.

#### Default Value

```ts
false
```

#### Inherited from

[`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md).[`extraction`](/docs/api/typescript/MemoryStoreConfig/index.md#extraction)

---

### writable

```ts
readonly writable: boolean;
```

Defined in: [src/memory/types.ts:108](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L108)

Whether this store accepts writes.

-   `false`: searchable only; never written to.
-   `true`: searchable and writable. Requires at least one write sink — `add`, `addMessages`, or both — to be implemented.

#### Overrides

[`MemoryStoreConfig`](/docs/api/typescript/MemoryStoreConfig/index.md).[`writable`](/docs/api/typescript/MemoryStoreConfig/index.md#writable)

## Methods

### search()

```ts
search(query, options?): Promise<MemoryEntry[]>;
```

Defined in: [src/memory/types.ts:110](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L110)

Search the store for entries matching the query, ordered by relevance.

#### Parameters

| Parameter | Type |
| --- | --- |
| `query` | `string` |
| `options?` | [`SearchOptions`](/docs/api/typescript/SearchOptions/index.md) |

#### Returns

`Promise`<[`MemoryEntry`](/docs/api/typescript/MemoryEntry/index.md)\[\]>

---

### add()?

```ts
optional add(content, metadata?): Promise<unknown>;
```

Defined in: [src/memory/types.ts:127](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L127)

Add a single piece of content to the store. Used by the `add_memory` tool, the programmatic MemoryManager.add, and by extraction when an [ExtractionConfig.extractor](/docs/api/typescript/ExtractionConfig/index.md#extractor) produces discrete entries (an extraction config with an extractor requires this method).

A store satisfies `writable: true` with `add`, [addMessages](#addmessages), or both. A store may also implement `add` while declaring `writable: false`, in which case it is never invoked.

Extraction writes are at-least-once: if one entry in a batch fails, the whole batch is retried, so `add` may be called again with content it already stored. Implementations used with extraction should tolerate duplicate writes (e.g. dedupe, or accept that retries may re-store an entry).

The resolved value is store-specific (e.g. a created record id or a write receipt) — each backend may return whatever shape fits it. The [MemoryManager](/docs/api/typescript/MemoryManager/index.md) does not consume this value (it only awaits completion); callers using a store directly can read it.

#### Parameters

| Parameter | Type |
| --- | --- |
| `content` | `string` |
| `metadata?` | `Record`<`string`, [`JSONValue`](/docs/api/typescript/JSONValue/index.md)\> |

#### Returns

`Promise`<`unknown`\>

---

### addMessages()?

```ts
optional addMessages(messages, context?): Promise<unknown>;
```

Defined in: [src/memory/types.ts:147](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L147)

Ingest a batch of conversation messages, preserving their role structure. This is the sink for automatic extraction that does not distill facts client-side: the manager hands the filtered [MessageData](/docs/api/typescript/MessageData/index.md) batch straight here in one call — no serialization, no model call. Backends that turn raw turns into memory themselves (e.g. role-aware conversational APIs that summarize server-side) implement this so the user/assistant structure survives. A store using extraction implements this method, unless it configures an [ExtractionConfig.extractor](/docs/api/typescript/ExtractionConfig/index.md#extractor) (which produces discrete entries written via [add](#add) instead).

Satisfies `writable: true` the same way [add](#add) does. The resolved value is store-specific and not consumed by the manager.

A store scopes its writes (e.g. by tenant or namespace) through its own configuration. The [AddMessagesContext](/docs/api/typescript/AddMessagesContext/index.md) parameter lets the manager pass additional per-batch context to the store.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`MessageData`](/docs/api/typescript/MessageData/index.md)\[\] | The filtered messages to ingest, in order |
| `context?` | [`AddMessagesContext`](/docs/api/typescript/AddMessagesContext/index.md) | Manager-supplied per-batch context (see [AddMessagesContext](/docs/api/typescript/AddMessagesContext/index.md)) |

#### Returns

`Promise`<`unknown`\>

---

### initialize()?

```ts
optional initialize(): Promise<void>;
```

Defined in: [src/memory/types.ts:154](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L154)

Perform async setup that must succeed before the agent runs.

Called by the [MemoryManager](/docs/api/typescript/MemoryManager/index.md) during `initAgent`. Stores that require remote resources (e.g. resolving a knowledge base type) implement this; the default is a no-op.

#### Returns

`Promise`<`void`\>

---

### getTools()?

```ts
optional getTools(): Tool[];
```

Defined in: [src/memory/types.ts:163](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/memory/types.ts#L163)

Returns store-specific tools to register with the agent, through a [MemoryManager](/docs/api/typescript/MemoryManager/index.md). Registers tools alongside `search_memory` / `add_memory` tools if enabled on the [MemoryManager](/docs/api/typescript/MemoryManager/index.md). Implement to expose backend-specific capabilities (e.g. a store-native query tool). Optional, mirrors [Plugin.getTools](/docs/api/typescript/Plugin/index.md#gettools).

#### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)\[\]

Array of tools provided by this store