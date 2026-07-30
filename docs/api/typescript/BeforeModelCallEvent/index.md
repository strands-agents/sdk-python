Defined in: [src/hooks/events.ts:382](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L382)

Event triggered just before the model is invoked. Fired before sending messages to the model for inference.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new BeforeModelCallEvent(data): BeforeModelCallEvent;
```

Defined in: [src/hooks/events.ts:403](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L403)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `model`: [`Model`](/docs/api/typescript/Model/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); `projectedInputTokens?`: `number`; } |
| `data.agent` | `LocalAgent` |
| `data.model` | [`Model`](/docs/api/typescript/Model/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |
| `data.projectedInputTokens?` | `number` |

#### Returns

`BeforeModelCallEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "beforeModelCallEvent";
```

Defined in: [src/hooks/events.ts:383](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L383)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:384](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L384)

---

### model

```ts
readonly model: Model;
```

Defined in: [src/hooks/events.ts:385](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L385)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:386](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L386)

---

### cancel

```ts
cancel: string | boolean = false;
```

Defined in: [src/hooks/events.ts:393](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L393)

Set by hook callbacks to cancel this model call. When set to `true`, a default cancel message is used. When set to a string, that string is used as the assistant response message.

---

### projectedInputTokens?

```ts
readonly optional projectedInputTokens?: number;
```

Defined in: [src/hooks/events.ts:401](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L401)

Projected input token count for the upcoming model call. Computed by the agent loop from message metadata and token estimation. Available for hooks and plugins (e.g. conversation managers) to make proactive decisions about context management.

## Methods

### toJSON()

```ts
toJSON(): Pick<BeforeModelCallEvent, "type" | "projectedInputTokens">;
```

Defined in: [src/hooks/events.ts:422](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/hooks/events.ts#L422)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`BeforeModelCallEvent`, `"type"` | `"projectedInputTokens"`\>