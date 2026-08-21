Defined in: [src/memory/types.ts:228](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L228)

Configuration for memory context injection.

When enabled on a [MemoryManager](/docs/api/typescript/MemoryManager/index.md), the manager searches memory before a model call and makes the top results available to the model for that call, so relevant knowledge is present without the model choosing to search. The injected text is ephemeral: it augments the model input for that call only and never persists into the durable conversation or session.

Extends the generic [InjectionConfig](/docs/api/typescript/InjectionConfig/index.md) (which carries `trigger`) with the memory-owned knobs: how many entries to retrieve, how to derive the query, and how to render the results.

## Extends

-   [`InjectionConfig`](/docs/api/typescript/InjectionConfig/index.md)

## Properties

### trigger?

```ts
optional trigger?:
  | InjectionTrigger
  | ((context) => boolean);
```

Defined in: [src/injection/types.ts:43](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/injection/types.ts#L43)

When injection runs. An [InjectionTrigger](/docs/api/typescript/InjectionTrigger/index.md) name selects a built-in policy; a predicate is the escape hatch — it receives the [InjectionContext](/docs/api/typescript/InjectionContext/index.md) and returns whether to inject this call. A predicate that throws fails open (injection is skipped, the model call proceeds).

#### Default Value

```ts
'userTurn'
```

#### Inherited from

[`InjectionConfig`](/docs/api/typescript/InjectionConfig/index.md).[`trigger`](/docs/api/typescript/InjectionConfig/index.md#trigger)

---

### maxEntries?

```ts
optional maxEntries?: number;
```

Defined in: [src/memory/types.ts:241](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L241)

Maximum number of entries to retrieve and inject per model call.

A store ranks by semantic similarity, which is not the same as contextual usefulness, so the default injects a small candidate set rather than betting on the top hit. Raising this improves recall at the cost of a larger prepend (context bloat); lower it for a tighter injection.

With multiple stores, results are concatenated in store-registration order with no cross-store ranking, so this cap can favor entries from earlier-registered stores.

#### Default Value

```ts
5
```

---

### query?

```ts
optional query?: (context) => string;
```

Defined in: [src/memory/types.ts:249](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L249)

Derives the search query from the current conversation. Return `undefined` or an empty string to skip injection for this call. A callback that throws fails open (injection is skipped).

Defaults to an adaptive query: the latest user message’s text on a user turn, otherwise the most recent assistant message’s text (the previous step on an autonomous turn).

#### Parameters

| Parameter | Type |
| --- | --- |
| `context` | { `messages`: [`MessageData`](/docs/api/typescript/MessageData/index.md)\[\]; } |
| `context.messages` | [`MessageData`](/docs/api/typescript/MessageData/index.md)\[\] |

#### Returns

`string`

---

### format?

```ts
optional format?: (context) => string;
```

Defined in: [src/memory/types.ts:259](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/memory/types.ts#L259)

Renders retrieved entries into the injected text. A callback that throws fails open (injection is skipped).

Defaults to a `<memory>` XML block with one `<entry>` per result, carrying a `source` attribute naming the originating store (when known) so the model can attribute and weigh each memory. The default escapes entry content and source, so a custom `format` that emits markup is responsible for its own escaping.

#### Parameters

| Parameter | Type |
| --- | --- |
| `context` | { `entries`: [`MemoryEntry`](/docs/api/typescript/MemoryEntry/index.md)\[\]; } |
| `context.entries` | [`MemoryEntry`](/docs/api/typescript/MemoryEntry/index.md)\[\] |

#### Returns

`string`