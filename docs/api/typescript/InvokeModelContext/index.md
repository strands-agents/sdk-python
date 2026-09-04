Defined in: [src/middleware/stages.ts:75](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L75)

Context passed to model-stage middleware. All inputs to the model call are explicit — middleware can inspect and transform any of them by passing a modified context to next(). Collection fields are defensive copies; invocationState and model are shared references.

## Properties

### agent

```ts
readonly agent: LocalAgent;
```

Defined in: [src/middleware/stages.ts:77](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L77)

The agent instance (escape hatch for advanced use cases).

---

### model

```ts
readonly model: Model;
```

Defined in: [src/middleware/stages.ts:79](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L79)

The model this call invokes. Initialized from agent.model and replaceable per call.

---

### messages

```ts
readonly messages: readonly Message[];
```

Defined in: [src/middleware/stages.ts:81](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L81)

The messages to send to the model.

---

### systemPrompt?

```ts
readonly optional systemPrompt?: SystemPrompt;
```

Defined in: [src/middleware/stages.ts:83](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L83)

System prompt to guide the model’s behavior.

---

### toolSpecs

```ts
readonly toolSpecs: readonly ToolSpec[];
```

Defined in: [src/middleware/stages.ts:85](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L85)

Tool specifications available to the model.

---

### toolChoice?

```ts
readonly optional toolChoice?: ToolChoice;
```

Defined in: [src/middleware/stages.ts:87](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L87)

Controls how the model selects tools.

---

### invocationState

```ts
readonly invocationState: InvocationState;
```

Defined in: [src/middleware/stages.ts:89](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L89)

Per-invocation state. Shared by reference — mutations are visible to hooks, tools, and AgentResult.

---

### projectedInputTokens?

```ts
readonly optional projectedInputTokens?: number;
```

Defined in: [src/middleware/stages.ts:91](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L91)

Estimated input token count for this model call, or undefined if estimation failed.

---

### dynamicTrailingBlocks?

```ts
readonly optional dynamicTrailingBlocks?: number;
```

Defined in: [src/middleware/stages.ts:100](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/middleware/stages.ts#L100)

How many trailing blocks of the last user message are rebuilt on every call.

Producers add to this; a provider placing cache points keeps its own ahead of the count, since a prefix that changes every call is never read back. Counted from the end of the message so it survives a provider’s content cleaning, which only drops earlier blocks.