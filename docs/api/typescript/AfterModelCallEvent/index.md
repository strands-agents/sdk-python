Defined in: [src/hooks/events.ts:467](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L467)

Event triggered after the model invocation completes. Fired after the model finishes generating a response, whether successful or failed. Uses reverse callback ordering for proper cleanup semantics.

Note: stopData may be undefined if an error occurs before the model completes.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new AfterModelCallEvent(data): AfterModelCallEvent;
```

Defined in: [src/hooks/events.ts:492](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L492)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `model`: [`Model`](/docs/api/typescript/Model/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); `attemptCount`: `number`; `stopData?`: [`ModelStopResponse`](/docs/api/typescript/ModelStopResponse/index.md); `error?`: `Error`; } |
| `data.agent` | `LocalAgent` |
| `data.model` | [`Model`](/docs/api/typescript/Model/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |
| `data.attemptCount` | `number` |
| `data.stopData?` | [`ModelStopResponse`](/docs/api/typescript/ModelStopResponse/index.md) |
| `data.error?` | `Error` |

#### Returns

`AfterModelCallEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "afterModelCallEvent";
```

Defined in: [src/hooks/events.ts:468](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L468)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:469](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L469)

---

### model

```ts
readonly model: Model;
```

Defined in: [src/hooks/events.ts:470](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L470)

---

### stopData?

```ts
readonly optional stopData?: ModelStopResponse;
```

Defined in: [src/hooks/events.ts:471](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L471)

---

### error?

```ts
readonly optional error?: Error;
```

Defined in: [src/hooks/events.ts:472](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L472)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:473](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L473)

---

### attemptCount

```ts
readonly attemptCount: number;
```

Defined in: [src/hooks/events.ts:484](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L484)

1-indexed count of model attempts for this turn, including the attempt that just completed (or failed). The first call in a turn is `1`; each subsequent retry increments by one.

Retry strategies may rely on `attemptCount === 1` to mark the start of a new retry sequence (e.g. to clear per-turn state carried over from a previous turn). The agent loop guarantees this marker on every fresh turn.

---

### retry?

```ts
optional retry?: boolean;
```

Defined in: [src/hooks/events.ts:490](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L490)

Optional flag that can be set by hook callbacks to request a retry of the model call. When set to true, the agent will retry the model invocation.

## Methods

### toJSON()

```ts
toJSON(): Pick<AfterModelCallEvent, "type" | "stopData" | "attemptCount"> & {
  error?: {
     message?: string;
  };
};
```

Defined in: [src/hooks/events.ts:522](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/hooks/events.ts#L522)

Serializes for wire transport, excluding the agent reference, invocationState, and mutable retry flag. Converts Error to an extensible object for safe wire serialization. Called automatically by JSON.stringify().

#### Returns

`Pick`<`AfterModelCallEvent`, `"type"` | `"stopData"` | `"attemptCount"`\> & { `error?`: { `message?`: `string`; }; }