Defined in: [src/agent/agent-as-tool.ts:20](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/agent/agent-as-tool.ts#L20)

Options for creating an agent tool via [Agent.asTool](/docs/api/typescript/Agent/index.md#astool).

## Properties

### name?

```ts
optional name?: string;
```

Defined in: [src/agent/agent-as-tool.ts:28](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/agent/agent-as-tool.ts#L28)

Tool name exposed to the parent agent’s model. Must match the pattern `[a-zA-Z0-9_-]{1,64}`.

Defaults to the agent’s name. Throws if the resolved name is not a valid tool name — provide an explicit name option to override.

---

### description?

```ts
optional description?: string;
```

Defined in: [src/agent/agent-as-tool.ts:37](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/agent/agent-as-tool.ts#L37)

Tool description exposed to the parent agent’s model. Helps the model understand when to use this tool.

Defaults to the agent’s description, or a generic description if the agent has no description set.

---

### preserveContext?

```ts
optional preserveContext?: boolean;
```

Defined in: [src/agent/agent-as-tool.ts:51](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/agent/agent-as-tool.ts#L51)

Whether to preserve the agent’s conversation history across invocations.

When `false` (default), the agent’s messages and state are reset to the values they had at the time the tool was created, ensuring every call starts from the same baseline.

When `true`, the agent retains its conversation history across invocations, allowing it to build context over multiple calls.

#### Default Value

```ts
false
```