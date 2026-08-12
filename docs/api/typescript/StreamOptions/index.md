Defined in: [src/models/model.ts:200](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L200)

Options interface for configuring streaming model invocation.

## Properties

### cancelSignal?

```ts
optional cancelSignal?: AbortSignal;
```

Defined in: [src/models/model.ts:205](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L205)

Optional cancellation signal that a provider implementation can forward to abort an in-flight request. Support is provider-dependent.

---

### systemPrompt?

```ts
optional systemPrompt?: SystemPrompt;
```

Defined in: [src/models/model.ts:211](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L211)

System prompt to guide the model’s behavior. Can be a simple string or an array of content blocks for advanced caching.

---

### toolSpecs?

```ts
optional toolSpecs?: ToolSpec[];
```

Defined in: [src/models/model.ts:216](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L216)

Array of tool specifications that the model can use.

---

### toolChoice?

```ts
optional toolChoice?: ToolChoice;
```

Defined in: [src/models/model.ts:221](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L221)

Controls how the model selects tools to use.

---

### modelState?

```ts
optional modelState?: StateStore;
```

Defined in: [src/models/model.ts:229](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/model.ts#L229)

Runtime state for model providers that manage server-side conversation state. The model can read and write this state during streaming (e.g., to store a response ID for conversation chaining). Mutations via `set`/`delete` are visible to the caller after the stream completes.