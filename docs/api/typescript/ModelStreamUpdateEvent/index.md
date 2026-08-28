Defined in: [src/hooks/events.ts:539](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L539)

Event triggered for each streaming event from the model. Wraps a [ModelStreamEvent](/docs/api/typescript/ModelStreamEvent/index.md) (transient streaming delta) during model inference. Completed content blocks are handled separately by [ContentBlockEvent](/docs/api/typescript/ContentBlockEvent/index.md) because they represent different granularities: partial deltas vs fully assembled results.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new ModelStreamUpdateEvent(data): ModelStreamUpdateEvent;
```

Defined in: [src/hooks/events.ts:545](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L545)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `event`: [`ModelStreamEvent`](/docs/api/typescript/ModelStreamEvent/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.event` | [`ModelStreamEvent`](/docs/api/typescript/ModelStreamEvent/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`ModelStreamUpdateEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "modelStreamUpdateEvent";
```

Defined in: [src/hooks/events.ts:540](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L540)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:541](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L541)

---

### event

```ts
readonly event: ModelStreamEvent;
```

Defined in: [src/hooks/events.ts:542](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L542)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:543](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L543)

## Methods

### toJSON()

```ts
toJSON(): Pick<ModelStreamUpdateEvent, "type" | "event">;
```

Defined in: [src/hooks/events.ts:556](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/hooks/events.ts#L556)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`ModelStreamUpdateEvent`, `"type"` | `"event"`\>