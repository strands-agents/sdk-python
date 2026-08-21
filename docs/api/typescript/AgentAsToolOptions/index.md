Defined in: [src/agent/agent-as-tool.ts:27](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/agent/agent-as-tool.ts#L27)

Options for creating an agent tool via [Agent.asTool](/docs/api/typescript/Agent/index.md#astool).

## Properties

### name?

```ts
optional name?: string;
```

Defined in: [src/agent/agent-as-tool.ts:35](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/agent/agent-as-tool.ts#L35)

Tool name exposed to the parent agent’s model. Must match the pattern `[a-zA-Z0-9_-]{1,64}`.

Defaults to the agent’s name. Throws if the resolved name is not a valid tool name — provide an explicit name option to override.

---

### description?

```ts
optional description?: string;
```

Defined in: [src/agent/agent-as-tool.ts:44](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/agent/agent-as-tool.ts#L44)

Tool description exposed to the parent agent’s model. Helps the model understand when to use this tool.

Defaults to the agent’s description, or a generic description if the agent has no description set.

---

### preserveContext?

```ts
optional preserveContext?: boolean;
```

Defined in: [src/agent/agent-as-tool.ts:58](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/agent/agent-as-tool.ts#L58)

Whether to preserve the agent’s conversation history across invocations.

When `false` (default), the agent’s messages and state are reset to the values they had at the time the tool was created, ensuring every call starts from the same baseline.

When `true`, the agent retains its conversation history across invocations, allowing it to build context over multiple calls.

#### Default Value

```ts
false
```

---

### delegate?

```ts
optional delegate?: boolean;
```

Defined in: [src/agent/agent-as-tool.ts:69](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/agent/agent-as-tool.ts#L69)

When true, the orchestrator treats this tool’s result as the final response and exits without an additional model call.

A delegation tool’s description is automatically suffixed with an instruction telling the model that this tool should be the only tool called in the turn.

#### Default Value

```ts
false
```