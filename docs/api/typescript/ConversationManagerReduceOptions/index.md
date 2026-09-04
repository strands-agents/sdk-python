```ts
type ConversationManagerReduceOptions = {
  agent: LocalAgent;
  model: Model;
  error?: ContextWindowOverflowError;
};
```

Defined in: [src/conversation-manager/conversation-manager.ts:27](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/conversation-manager/conversation-manager.ts#L27)

Options passed to [ConversationManager.reduce](/docs/api/typescript/ConversationManager/index.md#reduce).

When `error` is set, this is a reactive overflow recovery call — the implementation MUST remove enough history for the next model call to succeed.

When `error` is undefined, this is a proactive compression call — best-effort reduction to avoid hitting the context window limit.

## Properties

### agent

```ts
agent: LocalAgent;
```

Defined in: [src/conversation-manager/conversation-manager.ts:31](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/conversation-manager/conversation-manager.ts#L31)

The agent instance. Mutate `agent.messages` in place to reduce history.

---

### model

```ts
model: Model;
```

Defined in: [src/conversation-manager/conversation-manager.ts:37](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/conversation-manager/conversation-manager.ts#L37)

The model instance. Used by conversation managers that perform model-based reduction (e.g. summarization).

---

### error?

```ts
optional error?: ContextWindowOverflowError;
```

Defined in: [src/conversation-manager/conversation-manager.ts:49](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/conversation-manager/conversation-manager.ts#L49)

The [ContextWindowOverflowError](/docs/api/typescript/ContextWindowOverflowError/index.md) that triggered this call, or `undefined` for proactive compression calls.

When set, `reduce` MUST remove enough history for the next model call to succeed, or this error will propagate out of the agent loop uncaught.

When undefined, `reduce` is best-effort — errors are swallowed and the model call proceeds regardless.