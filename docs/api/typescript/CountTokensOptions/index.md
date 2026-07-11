Defined in: [src/models/model.ts:159](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/model.ts#L159)

Options for counting tokens in a set of messages.

## Properties

### systemPrompt?

```ts
optional systemPrompt?: SystemPrompt;
```

Defined in: [src/models/model.ts:164](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/model.ts#L164)

System prompt to guide the model’s behavior. Can be a simple string or an array of content blocks for advanced caching.

---

### toolSpecs?

```ts
optional toolSpecs?: ToolSpec[];
```

Defined in: [src/models/model.ts:169](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/model.ts#L169)

Array of tool specifications to include in the count.