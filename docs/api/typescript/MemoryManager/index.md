Defined in: [src/memory/memory-manager.ts:87](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L87)

Provides cross-session memory retrieval and storage for agents.

Manages one or more [MemoryStore](/docs/api/typescript/MemoryStore/index.md) backends, exposing `search_memory` and `add_memory` tools for agent-driven recall and persistence. Any tools the stores themselves provide (via [MemoryStore.getTools](/docs/api/typescript/MemoryStore/index.md#gettools)) are registered alongside these.

## Example

```typescript
import { Agent, MemoryManager } from '@strands-agents/sdk'

// Config shorthand
const agent = new Agent({
  model,
  memoryManager: { stores: [myStore], addToolConfig: true },
})

// Class instance (for programmatic access)
const memoryManager = new MemoryManager({ stores: [myStore], addToolConfig: true })
const agent = new Agent({ model, memoryManager })
await memoryManager.search('user preferences')
```

## Implements

-   [`Plugin`](/docs/api/typescript/Plugin/index.md)

## Constructors

### Constructor

```ts
new MemoryManager(config): MemoryManager;
```

Defined in: [src/memory/memory-manager.ts:110](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L110)

#### Parameters

| Parameter | Type |
| --- | --- |
| `config` | [`MemoryManagerConfig`](/docs/api/typescript/MemoryManagerConfig/index.md) |

#### Returns

`MemoryManager`

## Properties

### name

```ts
readonly name: "strands:memory-manager" = 'strands:memory-manager';
```

Defined in: [src/memory/memory-manager.ts:88](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L88)

A stable string identifier for the plugin. Used for logging, duplicate detection, and plugin management.

For strands-vended plugins, names should be prefixed with `strands:`.

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`name`](/docs/api/typescript/Plugin/index.md#name)

## Methods

### initAgent()

```ts
initAgent(agent): Promise<void>;
```

Defined in: [src/memory/memory-manager.ts:220](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L220)

Initializes the plugin with the agent.

Wires up two independent behaviors:

-   **Extraction**: for any store configured with [ExtractionConfig](/docs/api/typescript/ExtractionConfig/index.md), buffers conversation messages and attaches each store’s triggers. A no-op when no store uses extraction.
-   **Injection**: when enabled, registers an `InvokeModelStage` middleware that folds retrieved memory into the model input for each call without touching durable history. See \_provideMemoryContext, the `renderContent` callback the middleware invokes.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent this plugin is being attached to |

#### Returns

`Promise`<`void`\>

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`initAgent`](/docs/api/typescript/Plugin/index.md#initagent)

---

### flush()

```ts
flush(): Promise<void>;
```

Defined in: [src/memory/memory-manager.ts:346](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L346)

Saves every store’s remaining messages and waits for all saves to finish. No-op when no store has extraction configured.

Extraction normally runs in the background, so the most recent turn may not be saved yet when the agent responds. Call this once at a boundary you control - typically your app’s shutdown handler - so nothing is lost. A process killed before then (crash, hard timeout) may still lose the last unsaved turn; a more frequent trigger narrows that window.

Do not call this after every turn alongside a periodic trigger: it forces a save each time and so defeats the trigger’s schedule.

#### Returns

`Promise`<`void`\>

---

### getTools()

```ts
getTools(): Tool[];
```

Defined in: [src/memory/memory-manager.ts:415](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L415)

Returns tools registered by this plugin.

Includes the manager’s own `search_memory` / `add_memory` tools (per their config) plus any tools the configured stores expose via [MemoryStore.getTools](/docs/api/typescript/MemoryStore/index.md#gettools).

#### Returns

[`Tool`](/docs/api/typescript/Tool/index.md)\[\]

Array of tools to register with the agent

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`getTools`](/docs/api/typescript/Plugin/index.md#gettools)

---

### search()

```ts
search(query, options?): Promise<MemoryEntry[]>;
```

Defined in: [src/memory/memory-manager.ts:451](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/memory-manager.ts#L451)

Search stores for entries matching the query. If `stores` is provided, only searches to those named stores.

This method is unscoped with full access to all configured stores. Tool-level store scoping is applied by the search tool callback. When `options.stores` is omitted, all stores are searched.

Only `maxSearchResults` and routing (`stores`) cross this layer. Store-specific search parameters (e.g. a Bedrock metadata `filter` or search-type override) are not expressible here across heterogeneous stores — set them as per-instance defaults on the store, or call the store’s own `search()` directly for full control. Per-instance store policy (such as a tenant filter) always applies, including when reached through the `search_memory` tool.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `query` | `string` | The search query string |
| `options?` | [`MemorySearchOptions`](/docs/api/typescript/MemorySearchOptions/index.md) | Optional max results (forwarded to all stores) and store name filter |

#### Returns

`Promise`<[`MemoryEntry`](/docs/api/typescript/MemoryEntry/index.md)\[\]>

Array of memory entries from matching stores