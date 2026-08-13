Defined in: [src/conversation-manager/conversation-manager.ts:106](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L106)

Abstract base class for conversation history management strategies.

The primary responsibility of a ConversationManager is overflow recovery: when the model returns a [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md), [ConversationManager.reduce](#reduce) is called with `error` set and MUST reduce the history enough for the next model call to succeed. If `reduce` returns `false` (no reduction performed), the error propagates out of the agent loop uncaught. This makes `reduce` a critical operation — implementations must be able to make meaningful progress when called with `error` set.

Subclasses can enable proactive compression by passing `proactiveCompression` in the options object to the base constructor. When enabled, the base class registers a `BeforeModelCallEvent` hook that checks projected input tokens against the model’s context window limit and calls `reduce` (without `error`) when the threshold is exceeded.

## Example

```typescript
class Last10MessagesManager extends ConversationManager {
  readonly name = 'my:last-10-messages'

  reduce({ agent }: ReduceOptions): boolean {
    if (agent.messages.length <= 10) return false
    agent.messages.splice(0, agent.messages.length - 10)
    return true
  }
}
```

## Extended by

-   [`NullConversationManager`](/docs/api/typescript/NullConversationManager/index.md)
-   [`SlidingWindowConversationManager`](/docs/api/typescript/SlidingWindowConversationManager/index.md)
-   [`SummarizingConversationManager`](/docs/api/typescript/SummarizingConversationManager/index.md)

## Implements

-   [`Plugin`](/docs/api/typescript/Plugin/index.md)

## Constructors

### Constructor

```ts
new ConversationManager(options?): ConversationManager;
```

Defined in: [src/conversation-manager/conversation-manager.ts:117](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L117)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options?` | [`ConversationManagerOptions`](/docs/api/typescript/ConversationManagerOptions/index.md) | Configuration options for the conversation manager. |

#### Returns

`ConversationManager`

## Properties

### name

```ts
abstract readonly name: string;
```

Defined in: [src/conversation-manager/conversation-manager.ts:110](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L110)

A stable string identifier for this conversation manager.

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`name`](/docs/api/typescript/Plugin/index.md#name)

---

### \_compressionThreshold

```ts
protected readonly _compressionThreshold: number;
```

Defined in: [src/conversation-manager/conversation-manager.ts:112](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L112)

## Methods

### reduce()

```ts
abstract reduce(options): boolean | Promise<boolean>;
```

Defined in: [src/conversation-manager/conversation-manager.ts:149](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L149)

Reduce the conversation history.

Called in two scenarios:

1.  **Reactive** (error set): A [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md) occurred. The implementation MUST remove enough history for the next model call to succeed. Returning `false` means no reduction was possible, and the error will propagate out of the agent loop.
2.  **Proactive** (error undefined): The compression threshold was exceeded. This is best-effort — returning `false` or throwing is acceptable; the model call proceeds regardless.

Implementations should mutate `agent.messages` in place and return `true` if any reduction was performed, `false` otherwise.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`ConversationManagerReduceOptions`](/docs/api/typescript/ConversationManagerReduceOptions/index.md) | The reduction options |

#### Returns

`boolean` | `Promise`<`boolean`\>

`true` if the history was reduced, `false` otherwise. May return a `Promise` for implementations that need async I/O (e.g. model calls).

---

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/conversation-manager/conversation-manager.ts:166](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L166)

Initialize the conversation manager with the agent instance.

Registers two hooks:

1.  `AfterModelCallEvent`: Overflow recovery — when a [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md) occurs, calls [ConversationManager.reduce](#reduce) with `error` set and retries if reduction succeeded.
2.  `BeforeModelCallEvent`: Proactive compression — when projected input tokens exceed the configured compression threshold, calls [ConversationManager.reduce](#reduce) without `error`. The hook is always registered but only acts when proactive compression is enabled.

Subclasses that override `initAgent` MUST call `super.initAgent(agent)` to preserve overflow recovery and proactive compression behavior.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent to register hooks with |

#### Returns

`void`

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`initAgent`](/docs/api/typescript/Plugin/index.md#initagent)