```ts
type AgentConfig = {
  model?:   | Model<BaseModelConfig>
     | string;
  messages?:   | Message[]
     | MessageData[];
  tools?: ToolList;
  systemPrompt?:   | SystemPrompt
     | SystemPromptData;
  appState?: Record<string, JSONValue>;
  modelState?: Record<string, JSONValue>;
  printer?: boolean;
  conversationManager?: ConversationManager;
  contextManager?: ContextManagerStrategy;
  plugins?: Plugin[];
  retryStrategy?:   | RetryStrategy
     | RetryStrategy[]
     | null;
  interventions?: InterventionHandler[];
  structuredOutputSchema?: z.ZodSchema;
  sessionManager?: SessionManager;
  memoryManager?:   | MemoryManager
     | MemoryManagerConfig;
  traceAttributes?: Record<string, AttributeValue>;
  name?: string;
  description?: string;
  id?: string;
  toolExecutor?:   | ConcurrentToolExecutor
     | SequentialToolExecutor
     | ToolExecutorStrategy;
  checkpointing?: boolean;
  sandbox?: Sandbox | false;
};
```

Defined in: [src/agent/agent.ts:170](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L170)

Configuration object for creating a new Agent.

## Properties

### model?

```ts
optional model?:
  | Model<BaseModelConfig>
  | string;
```

Defined in: [src/agent/agent.ts:193](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L193)

The model instance that the agent will use to make decisions. Accepts either a Model instance or a string representing a Bedrock model ID. When a string is provided, it will be used to create a BedrockModel instance.

#### Example

```typescript
// Using a string model ID (creates BedrockModel)
const agent = new Agent({
  model: 'global.anthropic.claude-sonnet-4-6'
})

// Using an explicit BedrockModel instance with configuration
const agent = new Agent({
  model: new BedrockModel({
    modelId: 'global.anthropic.claude-sonnet-4-6',
    temperature: 0.7,
    maxTokens: 2048
  })
})
```

---

### messages?

```ts
optional messages?:
  | Message[]
  | MessageData[];
```

Defined in: [src/agent/agent.ts:195](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L195)

An initial set of messages to seed the agent’s conversation history.

---

### tools?

```ts
optional tools?: ToolList;
```

Defined in: [src/agent/agent.ts:201](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L201)

An initial set of tools to register with the agent. Accepts nested arrays of tools at any depth, which will be flattened automatically. [Agent](/docs/api/typescript/Agent/index.md) instances are automatically wrapped as tools via [Agent.asTool](/docs/api/typescript/Agent/index.md#astool).

---

### systemPrompt?

```ts
optional systemPrompt?:
  | SystemPrompt
  | SystemPromptData;
```

Defined in: [src/agent/agent.ts:205](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L205)

A system prompt which guides model behavior.

---

### appState?

```ts
optional appState?: Record<string, JSONValue>;
```

Defined in: [src/agent/agent.ts:207](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L207)

Optional initial state values for the agent.

---

### modelState?

```ts
optional modelState?: Record<string, JSONValue>;
```

Defined in: [src/agent/agent.ts:212](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L212)

Optional initial model-provider state (e.g., restoring `responseId` from a prior session). Typically only set when hydrating from a snapshot.

---

### printer?

```ts
optional printer?: boolean;
```

Defined in: [src/agent/agent.ts:218](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L218)

Enable automatic printing of agent output to console. When true, prints text generation, reasoning, and tool usage as they occur. Defaults to true.

---

### conversationManager?

```ts
optional conversationManager?: ConversationManager;
```

Defined in: [src/agent/agent.ts:223](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L223)

Conversation manager for handling message history and context overflow. Defaults to SlidingWindowConversationManager with windowSize of 40.

---

### contextManager?

```ts
optional contextManager?: ContextManagerStrategy;
```

Defined in: [src/agent/agent.ts:237](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L237)

Context management strategy.

-   `"auto"`: SummarizingConversationManager with proactive compression + ContextOffloader.
-   `"agentic"`: Lets the model drive context management via injected tools.

If `conversationManager` is also provided, the user’s conversation manager is used instead. Defaults to undefined (SlidingWindowConversationManager, no offloader).

#### Remarks

The offloader uses in-memory storage that does not persist across process restarts. For agents using `sessionManager`, provide an explicit `ContextOffloader` with durable storage via the `plugins` parameter.

---

### plugins?

```ts
optional plugins?: Plugin[];
```

Defined in: [src/agent/agent.ts:241](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L241)

Plugins to register with the agent.

---

### retryStrategy?

```ts
optional retryStrategy?:
  | RetryStrategy
  | RetryStrategy[]
  | null;
```

Defined in: [src/agent/agent.ts:252](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L252)

Retry strategy (or strategies) for failed model/tool calls.

-   Omitted: a sensible default [DefaultModelRetryStrategy](/docs/api/typescript/DefaultModelRetryStrategy/index.md) with exponential backoff is used.
-   Single strategy: the given strategy is used.
-   Array of strategies: all are registered, in the given order. Passing two instances of the same concrete class logs a warning — they will collide on `plugin.name` when the plugin registry initializes.
-   `null` or `[]`: retries are explicitly disabled; failures propagate to the caller.

---

### interventions?

```ts
optional interventions?: InterventionHandler[];
```

Defined in: [src/agent/agent.ts:256](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L256)

Intervention handlers evaluated in registration order at each lifecycle point.

---

### structuredOutputSchema?

```ts
optional structuredOutputSchema?: z.ZodSchema;
```

Defined in: [src/agent/agent.ts:260](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L260)

Zod schema for structured output validation.

---

### sessionManager?

```ts
optional sessionManager?: SessionManager;
```

Defined in: [src/agent/agent.ts:264](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L264)

Session manager for saving and restoring agent sessions

---

### memoryManager?

```ts
optional memoryManager?:
  | MemoryManager
  | MemoryManagerConfig;
```

Defined in: [src/agent/agent.ts:270](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L270)

Memory manager for cross-session memory retrieval and storage. Manages one or more memory stores and exposes search/add tools. Accepts a [MemoryManager](/docs/api/typescript/MemoryManager/index.md) instance or a [MemoryManagerConfig](/docs/api/typescript/MemoryManagerConfig/index.md) object (auto-wrapped).

---

### traceAttributes?

```ts
optional traceAttributes?: Record<string, AttributeValue>;
```

Defined in: [src/agent/agent.ts:276](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L276)

Custom trace attributes to include in all spans. These attributes are merged with standard attributes in telemetry spans. Telemetry must be enabled globally via telemetry.setupTracer() for these to take effect.

---

### name?

```ts
optional name?: string;
```

Defined in: [src/agent/agent.ts:280](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L280)

Optional name for the agent. Defaults to “Strands Agent”.

---

### description?

```ts
optional description?: string;
```

Defined in: [src/agent/agent.ts:284](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L284)

Optional description of what the agent does.

---

### id?

```ts
optional id?: string;
```

Defined in: [src/agent/agent.ts:288](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L288)

Optional unique identifier for the agent. Defaults to “agent”.

---

### toolExecutor?

```ts
optional toolExecutor?:
  | ConcurrentToolExecutor
  | SequentialToolExecutor
  | ToolExecutorStrategy;
```

Defined in: [src/agent/agent.ts:296](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L296)

Executor for tool calls from a single assistant turn.

Accepts a [ConcurrentToolExecutor](/docs/api/typescript/ConcurrentToolExecutor/index.md), a [SequentialToolExecutor](/docs/api/typescript/SequentialToolExecutor/index.md), or the corresponding [ToolExecutorStrategy](/docs/api/typescript/ToolExecutorStrategy/index.md) string shorthand. Defaults to concurrent execution.

---

### checkpointing?

```ts
optional checkpointing?: boolean;
```

Defined in: [src/agent/agent.ts:309](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L309)

**`Experimental`**

When `true`, the agent loop pauses at cycle boundaries (`afterModel`, `afterTools`) and returns `stopReason: 'checkpoint'` with a populated `checkpoint` field. Resume by passing the checkpoint back as `{ checkpointResume: { checkpoint: ... } }`.

The SDK does not capture conversation state in the checkpoint; pair with a `SessionManager` for cross-process state continuity. Defaults to `false`. See the experimental checkpoint module.

---

### sandbox?

```ts
optional sandbox?: Sandbox | false;
```

Defined in: [src/agent/agent.ts:323](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/agent/agent.ts#L323)

Execution environment for running commands, code, and file operations. When provided, sandbox-aware tools route operations through it.

Two distinct intents, even though they resolve to the same host execution in Node today:

-   Omitted: use the environment’s default sandbox (host execution in Node). This default is the slot reserved for richer behavior later.
-   `false`: explicitly opt out of a managed sandbox and run on the host.

Keep `false` distinct from omitting so the opt-out stays stable even if the default changes.