Defined in: [src/hooks/events.ts:571](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L571)

Event triggered when a content block completes during model inference. Wraps completed content blocks (TextBlock, ToolUseBlock, ReasoningBlock) from model streaming. This is intentionally separate from [ModelStreamUpdateEvent](/docs/api/typescript/ModelStreamUpdateEvent/index.md). The model’s `streamAggregated()` yields two kinds of output: [ModelStreamEvent](/docs/api/typescript/ModelStreamEvent/index.md) (transient streaming deltas — partial data arriving while the model generates) and [ContentBlock](/docs/api/typescript/ContentBlock/index.md) (fully assembled results after all deltas accumulate). These represent different granularities with different semantics, so they are wrapped in distinct event classes rather than combined into a single event.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new ContentBlockEvent(data): ContentBlockEvent;
```

Defined in: [src/hooks/events.ts:577](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L577)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `contentBlock`: [`ContentBlock`](/docs/api/typescript/ContentBlock/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.contentBlock` | [`ContentBlock`](/docs/api/typescript/ContentBlock/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`ContentBlockEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "contentBlockEvent";
```

Defined in: [src/hooks/events.ts:572](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L572)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:573](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L573)

---

### contentBlock

```ts
readonly contentBlock: ContentBlock;
```

Defined in: [src/hooks/events.ts:574](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L574)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:575](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L575)

## Methods

### toJSON()

```ts
toJSON(): Pick<ContentBlockEvent, "type" | "contentBlock">;
```

Defined in: [src/hooks/events.ts:588](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/hooks/events.ts#L588)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`ContentBlockEvent`, `"type"` | `"contentBlock"`\>