Defined in: [src/hooks/events.ts:732](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L732)

Event triggered before executing tools. Fired when the model returns tool use blocks that need to be executed. Hook callbacks can set [cancel](#cancel) to prevent all tools from executing.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Implements

-   `Interruptible`

## Constructors

### Constructor

```ts
new BeforeToolsEvent(data): BeforeToolsEvent;
```

Defined in: [src/hooks/events.ts:745](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L745)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `message`: [`Message`](/docs/api/typescript/Message/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.message` | [`Message`](/docs/api/typescript/Message/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`BeforeToolsEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "beforeToolsEvent";
```

Defined in: [src/hooks/events.ts:733](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L733)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:734](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L734)

---

### message

```ts
readonly message: Message;
```

Defined in: [src/hooks/events.ts:735](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L735)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:736](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L736)

---

### cancel

```ts
cancel: string | boolean = false;
```

Defined in: [src/hooks/events.ts:743](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L743)

Set by hook callbacks to cancel all tool calls. When set to `true`, a default cancel message is used. When set to a string, that string is used as the tool result error message.

## Methods

### interrupt()

```ts
interrupt<T>(params): T;
```

Defined in: [src/hooks/events.ts:760](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L760)

Raises an interrupt for human-in-the-loop workflows. If a response is available (from a previous resume), returns it immediately. Otherwise, throws an InterruptError to halt agent execution.

#### Type Parameters

| Type Parameter | Default type |
| --- | --- |
| `T` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) |

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `params` | [`InterruptParams`](/docs/api/typescript/InterruptParams/index.md) | Interrupt parameters including name and optional reason |

#### Returns

`T`

The user’s response when resuming from an interrupt

#### Implementation of

```ts
Interruptible.interrupt
```

---

### toJSON()

```ts
toJSON(): Pick<BeforeToolsEvent, "type" | "message">;
```

Defined in: [src/hooks/events.ts:768](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L768)

Serializes for wire transport, excluding the agent reference, invocationState, and mutable cancel flag. Called automatically by JSON.stringify().

#### Returns

`Pick`<`BeforeToolsEvent`, `"type"` | `"message"`\>