Defined in: [src/middleware/stages.ts:74](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L74)

Context passed to model-stage middleware. All inputs to the model call are explicit — middleware can inspect and transform any of them by passing a modified context to next().

## Properties

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/middleware/stages.ts:76](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L76)

The agent instance (escape hatch for advanced use cases).

---

### messages

```ts
readonly messages: readonly Message[];
```

Defined in: [src/middleware/stages.ts:78](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L78)

The messages to send to the model.

---

### systemPrompt?

```ts
readonly optional systemPrompt?: SystemPrompt;
```

Defined in: [src/middleware/stages.ts:80](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L80)

System prompt to guide the model’s behavior.

---

### toolSpecs

```ts
readonly toolSpecs: readonly ToolSpec[];
```

Defined in: [src/middleware/stages.ts:82](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L82)

Tool specifications available to the model.

---

### toolChoice?

```ts
readonly optional toolChoice?: ToolChoice;
```

Defined in: [src/middleware/stages.ts:84](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L84)

Controls how the model selects tools.

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/middleware/stages.ts:86](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L86)

Per-invocation state. Shared by reference — mutations are visible to hooks, tools, and AgentResult.

---

### projectedInputTokens?

```ts
readonly optional projectedInputTokens?: number;
```

Defined in: [src/middleware/stages.ts:88](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/middleware/stages.ts#L88)

Estimated input token count for this model call, or undefined if estimation failed.