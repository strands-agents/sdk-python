Defined in: [src/conversation-manager/null-conversation-manager.ts:17](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/null-conversation-manager.ts#L17)

A no-op conversation manager that does not modify the conversation history.

Does not register any proactive hooks. Overflow errors will not be retried since `reduce` always returns `false`.

## Extends

-   [`ConversationManager`](/docs/api/typescript/ConversationManager/index.md)

## Constructors

### Constructor

```ts
new NullConversationManager(options?): NullConversationManager;
```

Defined in: [src/conversation-manager/conversation-manager.ts:121](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/conversation-manager.ts#L121)

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options?` | [`ConversationManagerOptions`](/docs/api/typescript/ConversationManagerOptions/index.md) | Configuration options for the conversation manager. |

#### Returns

`NullConversationManager`

#### Inherited from

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`constructor`](/docs/api/typescript/ConversationManager/index.md#constructor)

## Properties

### \_compressionThreshold

```ts
protected readonly _compressionThreshold: number;
```

Defined in: [src/conversation-manager/conversation-manager.ts:116](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/conversation-manager.ts#L116)

#### Inherited from

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`_compressionThreshold`](/docs/api/typescript/ConversationManager/index.md#_compressionthreshold)

---

### name

```ts
readonly name: "strands:null-conversation-manager" = 'strands:null-conversation-manager';
```

Defined in: [src/conversation-manager/null-conversation-manager.ts:21](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/null-conversation-manager.ts#L21)

Unique identifier for this conversation manager.

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`name`](/docs/api/typescript/ConversationManager/index.md#name)

## Methods

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/conversation-manager/conversation-manager.ts:170](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/conversation-manager.ts#L170)

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
reduce(_args): boolean;
```

Defined in: [src/conversation-manager/null-conversation-manager.ts:28](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/null-conversation-manager.ts#L28)

No-op reduction — never modifies the conversation history.

#### Parameters

| Parameter | Type |
| --- | --- |
| `_args` | [`ConversationManagerReduceOptions`](/docs/api/typescript/ConversationManagerReduceOptions/index.md) |

#### Returns

`boolean`

`false` always

#### Overrides

[`ConversationManager`](/docs/api/typescript/ConversationManager/index.md).[`reduce`](/docs/api/typescript/ConversationManager/index.md#reduce)