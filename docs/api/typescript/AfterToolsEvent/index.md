Defined in: [src/hooks/events.ts:778](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L778)

Event triggered after all tools complete execution. Fired after tool results are collected and ready to be added to conversation. Uses reverse callback ordering for proper cleanup semantics.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new AfterToolsEvent(data): AfterToolsEvent;
```

Defined in: [src/hooks/events.ts:798](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L798)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `message`: [`Message`](/docs/api/typescript/Message/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.message` | [`Message`](/docs/api/typescript/Message/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`AfterToolsEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "afterToolsEvent";
```

Defined in: [src/hooks/events.ts:779](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L779)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:780](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L780)

---

### message

```ts
readonly message: Message;
```

Defined in: [src/hooks/events.ts:781](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L781)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:782](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L782)

---

### endTurn

```ts
endTurn: string | boolean = false;
```

Defined in: [src/hooks/events.ts:796](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L796)

When set to `true`, the agent loop halts after this tool batch completes without calling the model again and a default message (`"Turn ended early by hook after tool execution"`) is appended as the final assistant message. When set to a string, that string is used instead of the default — the string becomes literal assistant content (a `TextBlock`), not a reason or label. Contrast with [cancel](/docs/api/typescript/BeforeToolCallEvent/index.md#cancel) fields on other events, where the string is a cancellation reason.

In both cases `stopReason` on the returned `AgentResult` is `'endTurn'`.

## Methods

### toJSON()

```ts
toJSON(): Pick<AfterToolsEvent, "type" | "message">;
```

Defined in: [src/hooks/events.ts:814](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/hooks/events.ts#L814)

Serializes for wire transport, excluding the agent reference, invocationState, and mutable endTurn field. Called automatically by JSON.stringify().

#### Returns

`Pick`<`AfterToolsEvent`, `"type"` | `"message"`\>