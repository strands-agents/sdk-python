Defined in: [src/hooks/events.ts:655](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L655)

Event triggered for each streaming progress event from a tool during execution. Wraps a [ToolStreamEvent](/docs/api/typescript/ToolStreamEvent/index.md) with agent context, keeping the tool authoring interface unchanged — tools construct `ToolStreamEvent` without knowledge of agents or hooks, and the agent layer wraps them at the boundary.

Consistent with [ModelStreamUpdateEvent](/docs/api/typescript/ModelStreamUpdateEvent/index.md) which wraps model streaming events the same way.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new ToolStreamUpdateEvent(data): ToolStreamUpdateEvent;
```

Defined in: [src/hooks/events.ts:661](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L661)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `event`: [`ToolStreamEvent`](/docs/api/typescript/ToolStreamEvent/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.event` | [`ToolStreamEvent`](/docs/api/typescript/ToolStreamEvent/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`ToolStreamUpdateEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "toolStreamUpdateEvent";
```

Defined in: [src/hooks/events.ts:656](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L656)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:657](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L657)

---

### event

```ts
readonly event: ToolStreamEvent;
```

Defined in: [src/hooks/events.ts:658](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L658)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:659](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L659)

## Methods

### toJSON()

```ts
toJSON(): Pick<ToolStreamUpdateEvent, "type" | "event">;
```

Defined in: [src/hooks/events.ts:672](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L672)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`ToolStreamUpdateEvent`, `"type"` | `"event"`\>