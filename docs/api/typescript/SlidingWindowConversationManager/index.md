Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:129](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L129)

Implements a sliding window strategy for managing conversation history.

This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids invalid window states. When the message count exceeds the window size, it will either truncate large tool results or remove the oldest messages while ensuring tool use/result pairs remain valid.

Registers hooks for:

-   AfterInvocationEvent: Applies sliding window management after each invocation
-   AfterModelCallEvent: Reduces context on overflow errors and requests retry (via super)
-   BeforeModelCallEvent: Proactive compression when threshold is exceeded (via super)

## Extends

-   [`ConversationManager`](/docs/api/typescript/ConversationManager/index.md)

## Constructors

### Constructor

```ts
new SlidingWindowConversationManager(config?): SlidingWindowConversationManager;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:145](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L145)

Initialize the sliding window conversation manager.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config?` | [`SlidingWindowConversationManagerConfig`](/docs/api/typescript/SlidingWindowConversationManagerConfig/index.md) | Configuration options for the sliding window manager. |

#### Returns

`SlidingWindowConversationManager`

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`constructor`](/docs/api/typescript/ConversationManager/index.md#constructor)

## Properties

### \_compressionThreshold

```ts
protected readonly _compressionThreshold: number;
```

Defined in: [src/conversation-manager/conversation-manager.ts:112](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L112)

#### Inherited from

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`_compressionThreshold`](/docs/api/typescript/ConversationManager/index.md#_compressionthreshold)

---

### name

```ts
readonly name: "strands:sliding-window-conversation-manager" = 'strands:sliding-window-conversation-manager';
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:138](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L138)

Unique identifier for this conversation manager.

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`name`](/docs/api/typescript/ConversationManager/index.md#name)

## Methods

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:162](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L162)

Initialize the plugin by registering hooks with the agent.

Registers:

-   AfterInvocationEvent callback to apply sliding window management
-   AfterModelCallEvent callback to handle context overflow and request retry (via super)
-   BeforeModelCallEvent callback for proactive compression (via super)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | The agent to register hooks with |

#### Returns

`void`

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`initAgent`](/docs/api/typescript/ConversationManager/index.md#initagent)

---

### reduce()

```ts
reduce(options): boolean;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:182](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L182)

Reduce the conversation history.

When `error` is set (reactive overflow recovery), attempts to truncate large tool results first before falling back to message trimming.

When `error` is undefined (proactive compression), only trims messages without attempting tool result truncation.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`ConversationManagerReduceOptions`](/docs/api/typescript/ConversationManagerReduceOptions/index.md) | The reduction options |

#### Returns

`boolean`

`true` if the history was reduced, `false` otherwise

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`reduce`](/docs/api/typescript/ConversationManager/index.md#reduce)