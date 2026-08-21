Defined in: [src/models/model.ts:209](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L209)

Options interface for configuring streaming model invocation.

## Properties

### cancelSignal?

```ts
optional cancelSignal?: AbortSignal;
```

Defined in: [src/models/model.ts:214](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L214)

Optional cancellation signal that a provider implementation can forward to abort an in-flight request. Support is provider-dependent.

---

### systemPrompt?

```ts
optional systemPrompt?: SystemPrompt;
```

Defined in: [src/models/model.ts:220](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L220)

System prompt to guide the model’s behavior. Can be a simple string or an array of content blocks for advanced caching.

---

### toolSpecs?

```ts
optional toolSpecs?: ToolSpec[];
```

Defined in: [src/models/model.ts:225](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L225)

Array of tool specifications that the model can use.

---

### toolChoice?

```ts
optional toolChoice?: ToolChoice;
```

Defined in: [src/models/model.ts:230](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L230)

Controls how the model selects tools to use.

---

### modelState?

```ts
optional modelState?: StateStore;
```

Defined in: [src/models/model.ts:238](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L238)

Runtime state for model providers that manage server-side conversation state. The model can read and write this state during streaming (e.g., to store a response ID for conversation chaining). Mutations via `set`/`delete` are visible to the caller after the stream completes.

---

### dynamicTrailingBlocks?

```ts
optional dynamicTrailingBlocks?: number;
```

Defined in: [src/models/model.ts:241](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/model.ts#L241)

How many trailing blocks of the last user message are rebuilt on every call.