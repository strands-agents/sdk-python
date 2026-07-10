Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:77](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L77)

Implements a summarization strategy for managing conversation history.

When a [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md) occurs, this manager summarizes the oldest messages using a model call and replaces them with a single summary message, preserving context that would otherwise be lost.

## Extends

-   [`ConversationManager`](/docs/api/typescript/ConversationManager/index.md)

## Constructors

### Constructor

```ts
new SummarizingConversationManager(config?): SummarizingConversationManager;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:87](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L87)

#### Parameters

| Parameter | Type |
| --- | --- |
| `config?` | [`SummarizingConversationManagerConfig`](/docs/api/typescript/SummarizingConversationManagerConfig/index.md) |

#### Returns

`SummarizingConversationManager`

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`constructor`](/docs/api/typescript/ConversationManager/index.md#constructor)

## Properties

### \_compressionThreshold

```ts
protected readonly _compressionThreshold: number;
```

Defined in: [src/conversation-manager/conversation-manager.ts:116](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/conversation-manager.ts#L116)

#### Inherited from

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`_compressionThreshold`](/docs/api/typescript/ConversationManager/index.md#_compressionthreshold)

---

### name

```ts
readonly name: "strands:summarizing-conversation-manager" = 'strands:summarizing-conversation-manager';
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:78](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L78)

A stable string identifier for this conversation manager.

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`name`](/docs/api/typescript/ConversationManager/index.md#name)

## Methods

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/conversation-manager/conversation-manager.ts:170](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/conversation-manager.ts#L170)

Initialize the conversation manager with the agent instance.

Registers two hooks:

1.  `AfterModelCallEvent`: Overflow recovery — when a [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md) occurs, calls [ConversationManager.reduce](/docs/api/typescript/ConversationManager/index.md#reduce) with `error` set and retries if reduction succeeded.
2.  `BeforeModelCallEvent`: Proactive compression — when projected input tokens exceed the configured compression threshold, calls [ConversationManager.reduce](/docs/api/typescript/ConversationManager/index.md#reduce) without `error`. The hook is always registered but only acts when proactive compression is enabled.

Subclasses that override `initAgent` MUST call `super.initAgent(agent)` to preserve overflow recovery and proactive compression behavior.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent to register hooks with |

#### Returns

`void`

#### Inherited from

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`initAgent`](/docs/api/typescript/ConversationManager/index.md#initagent)

---

### reduce()

```ts
reduce(options): Promise<boolean>;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:109](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L109)

Reduce the conversation history by summarizing older messages.

When `error` is set (reactive overflow recovery), summarization failure is rethrown with the original error as cause — the agent loop must not proceed with an overflow.

When `error` is undefined (proactive compression), summarization failure is logged and returns `false` — the model call proceeds regardless.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`ConversationManagerReduceOptions`](/docs/api/typescript/ConversationManagerReduceOptions/index.md) | The reduction options |

#### Returns

`Promise`<`boolean`\>

`true` if the history was reduced, `false` otherwise

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`reduce`](/docs/api/typescript/ConversationManager/index.md#reduce)