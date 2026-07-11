Defined in: [src/hooks/events.ts:708](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L708)

Event emitted when an interrupt is raised during agent execution. The `interrupt.source` field discriminates between tool-callback and hook-callback origins. One event fires per unanswered interrupt at the moment the agent stops to wait for responses.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new InterruptEvent(data): InterruptEvent;
```

Defined in: [src/hooks/events.ts:714](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L714)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `interrupt`: [`Interrupt`](/docs/api/typescript/Interrupt/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.interrupt` | [`Interrupt`](/docs/api/typescript/Interrupt/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`InterruptEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "interruptEvent";
```

Defined in: [src/hooks/events.ts:709](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L709)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:710](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L710)

---

### interrupt

```ts
readonly interrupt: Interrupt;
```

Defined in: [src/hooks/events.ts:711](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L711)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:712](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L712)

## Methods

### toJSON()

```ts
toJSON(): Pick<InterruptEvent, "type"> & {
  interrupt: {
     id: string;
     name: string;
     reason?: JSONValue;
     response?: JSONValue;
     source: InterruptSource;
  };
};
```

Defined in: [src/hooks/events.ts:722](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/hooks/events.ts#L722)

Serializes for wire transport, excluding agent and invocationState.

#### Returns

`Pick`<`InterruptEvent`, `"type"`\> & { `interrupt`: { `id`: `string`; `name`: `string`; `reason?`: [`JSONValue`](/docs/api/typescript/JSONValue/index.md); `response?`: [`JSONValue`](/docs/api/typescript/JSONValue/index.md); `source`: [`InterruptSource`](/docs/api/typescript/InterruptSource/index.md); }; }