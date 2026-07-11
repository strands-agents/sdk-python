Defined in: [src/hooks/events.ts:212](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L212)

Event triggered when the framework adds a message to the conversation history. Fired for user input, assistant responses, and tool-result messages added during agent execution. Does not fire for messages preloaded via `AgentConfig.messages` or messages manually pushed to `agent.messages`.

## Extends

-   [`HookableEvent`](/docs/api/typescript/HookableEvent/index.md)

## Constructors

### Constructor

```ts
new MessageAddedEvent(data): MessageAddedEvent;
```

Defined in: [src/hooks/events.ts:218](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L218)

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

Defined in: [src/hooks/events.ts:213](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L213)

---

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/hooks/events.ts:214](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L214)

---

### message

```ts
readonly message: Message;
```

Defined in: [src/hooks/events.ts:215](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L215)

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/hooks/events.ts:216](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L216)

## Methods

### toJSON()

```ts
toJSON(): Pick<MessageAddedEvent, "type" | "message">;
```

Defined in: [src/hooks/events.ts:229](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/hooks/events.ts#L229)

Serializes for wire transport, excluding the agent reference and invocationState. Called automatically by JSON.stringify().

#### Returns

`Pick`<`MessageAddedEvent`, `"type"` | `"message"`\>