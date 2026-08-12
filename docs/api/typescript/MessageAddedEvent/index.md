Defined in: [src/hooks/events.ts:213](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L213)

Event triggered when the framework adds a message to the conversation history. Fired for user input, assistant responses, and tool-result messages added during agent execution. Does not fire for messages preloaded via `AgentConfig.messages` or messages manually pushed to `agent.messages`.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new MessageAddedEvent(data): MessageAddedEvent;
```

Defined in: [src/hooks/events.ts:219](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L219)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `agent`: `LocalAgent`; `message`: [`Message`](/docs/api/typescript/Message/index.md); `invocationState`: [`InvocationState`](/docs/api/typescript/InvocationState/index.md); } |
| `data.agent` | `LocalAgent` |
| `data.message` | [`Message`](/docs/api/typescript/Message/index.md) |
| `data.invocationState` | [`InvocationState`](/docs/api/typescript/InvocationState/index.md) |

#### Returns

`MessageAddedEvent`

#### Overrides

[`HookableEvent`](/docs/api/typescript/HookableEvent/index.md).[`constructor`](/docs/api/typescript/HookableEvent/index.md#constructor)

## Properties

### type

```ts
readonly type: "messageAddedEvent";
```

Defined in: [src/hooks/events.ts:214](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L214)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:215](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L215)

---

### message

```ts
readonly message: Message;
```

Defined in: [src/hooks/events.ts:216](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L216)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:217](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L217)

## Methods

### toJSON()

```ts
toJSON(): Pick<MessageAddedEvent, "type" | "message">;
```

Defined in: [src/hooks/events.ts:230](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/hooks/events.ts#L230)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`MessageAddedEvent`, `"type"` | `"message"`\>