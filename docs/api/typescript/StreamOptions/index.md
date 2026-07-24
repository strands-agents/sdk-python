Defined in: [src/models/model.ts:130](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/model.ts#L130)

Options interface for configuring streaming model invocation.

## Properties

### systemPrompt?

```ts
optional systemPrompt?: SystemPrompt;
```

Defined in: [src/models/model.ts:135](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/model.ts#L135)

System prompt to guide the model’s behavior. Can be a simple string or an array of content blocks for advanced caching.

---

### toolSpecs?

```ts
optional toolSpecs?: ToolSpec[];
```

Defined in: [src/models/model.ts:140](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/model.ts#L140)

Array of tool specifications that the model can use.

---

### toolChoice?

```ts
optional toolChoice?: ToolChoice;
```

Defined in: [src/models/model.ts:145](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/model.ts#L145)

Controls how the model selects tools to use.

---

### modelState?

```ts
optional modelState?: StateStore;
```

Defined in: [src/models/model.ts:153](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/model.ts#L153)

Runtime state for model providers that manage server-side conversation state. The model can read and write this state during streaming (e.g., to store a response ID for conversation chaining). Mutations via `set`/`delete` are visible to the caller after the stream completes.