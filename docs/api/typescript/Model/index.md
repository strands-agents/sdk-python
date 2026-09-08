Defined in: [src/models/model.ts:301](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L301)

Base abstract class for model providers. Defines the contract that all model provider implementations must follow.

Model providers handle communication with LLM APIs and implement streaming responses using async iterables.

## Extended by

-   [`BedrockModel`](/docs/api/typescript/BedrockModel/index.md)

## Type Parameters

| Type Parameter | Default type | Description |
| --- | --- | --- |
| `T` *extends* [`BaseModelConfig`](/docs/api/typescript/BaseModelConfig/index.md) | [`BaseModelConfig`](/docs/api/typescript/BaseModelConfig/index.md) | Model configuration type extending BaseModelConfig |

## Constructors

### Constructor

```ts
new Model<T>(): Model<T>;
```

#### Returns

`Model`<`T`\>

## Accessors

### modelId

#### Get Signature

```ts
get modelId(): string;
```

Defined in: [src/models/model.ts:320](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L320)

The model ID from the current configuration, if configured.

##### Returns

`string`

---

### stateful

#### Get Signature

```ts
get stateful(): boolean;
```

Defined in: [src/models/model.ts:336](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L336)

Whether this model manages conversation state server-side.

When `true`, the server tracks conversation context across turns, so the SDK sends only the latest message instead of the full history. After each invocation, the agent’s local message history is cleared automatically.

Model providers that support server-side state management should override this to return `true`.

##### Returns

`boolean`

`false` by default

## Methods

### updateConfig()

```ts
abstract updateConfig(modelConfig): void;
```

Defined in: [src/models/model.ts:308](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L308)

Updates the model configuration. Merges the provided configuration with existing settings.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `modelConfig` | `T` | Configuration object with model-specific settings to update |

#### Returns

`void`

---

### getConfig()

```ts
abstract getConfig(): T;
```

Defined in: [src/models/model.ts:315](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L315)

Retrieves the current model configuration.

#### Returns

`T`

The current configuration object

---

### stream()

```ts
abstract stream(messages, options?): AsyncIterable<ModelStreamEvent>;
```

Defined in: [src/models/model.ts:348](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L348)

Streams a conversation with the model. Returns an async iterable that yields streaming events as they occur.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`Message`](/docs/api/typescript/Message/index.md)\[\] | Array of conversation messages |
| `options?` | [`StreamOptions`](/docs/api/typescript/StreamOptions/index.md) | Optional streaming configuration |

#### Returns

`AsyncIterable`<[`ModelStreamEvent`](/docs/api/typescript/ModelStreamEvent/index.md)\>

Async iterable of streaming events

---

### countTokens()

```ts
countTokens(messages, options?): Promise<number>;
```

Defined in: [src/models/model.ts:364](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L364)

Count tokens for the given input before sending to the model.

Used for proactive context management (e.g., triggering compression at a threshold). The base implementation uses a character-based heuristic (chars/4 for text, chars/2 for JSON).

Subclasses should override this method to use native token counting APIs (e.g., Bedrock CountTokens, Anthropic countTokens, Gemini countTokens) for improved accuracy, falling back to `super.countTokens()` on API failure.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`Message`](/docs/api/typescript/Message/index.md)\[\] | Array of conversation messages to count tokens for |
| `options?` | [`CountTokensOptions`](/docs/api/typescript/CountTokensOptions/index.md) | Optional options containing system prompt and tool specs |

#### Returns

`Promise`<`number`\>

Total input token count

---

### estimateUtilization()

```ts
estimateUtilization(inputTokens): number;
```

Defined in: [src/models/model.ts:377](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L377)

Estimate the fraction of the model’s context window consumed by the given input token count.

Resolves the model’s context window limit (falling back to DEFAULT\_CONTEXT\_WINDOW\_LIMIT with a warning when not configured) and returns `inputTokens / contextWindowLimit`.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `inputTokens` | `number` | Total input token count (e.g. from a model event’s projectedInputTokens) |

#### Returns

`number`

Token usage ratio (0–1+; above 1.0 means overflow)

---

### streamAggregated()

```ts
streamAggregated(messages, options?): AsyncGenerator<
  | ContentBlock
| ModelStreamEvent, StreamAggregatedResult, undefined>;
```

Defined in: [src/models/model.ts:444](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L444)

Streams a conversation with aggregated content blocks and messages. Returns an async generator that yields streaming events and content blocks, and returns the final message with stop reason and optional metadata.

This method enhances the basic stream() by collecting streaming events into complete ContentBlock and Message objects, which are needed by the agentic loop for tool execution and conversation management.

The method yields:

-   ModelStreamEvent - Original streaming events (passed through)
-   ContentBlock - Complete content block (emitted when block completes)

The method returns:

-   StreamAggregatedResult containing the complete message, stop reason, and optional metadata

All exceptions thrown from this method are wrapped in ModelError to provide a consistent error type for model-related errors. Specific error subtypes like ContextWindowOverflowError, ModelThrottledError, and MaxTokensError are preserved.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`Message`](/docs/api/typescript/Message/index.md)\[\] | Array of conversation messages |
| `options?` | [`StreamOptions`](/docs/api/typescript/StreamOptions/index.md) | Optional streaming configuration |

#### Returns

`AsyncGenerator`< | [`ContentBlock`](/docs/api/typescript/ContentBlock/index.md) | [`ModelStreamEvent`](/docs/api/typescript/ModelStreamEvent/index.md), `StreamAggregatedResult`, `undefined`\>

Async generator yielding ModelStreamEvent | ContentBlock and returning a StreamAggregatedResult

#### Throws

ModelError - Base class for all model-related errors

#### Throws

ContextWindowOverflowError - When input exceeds the model’s context window

#### Throws

ModelThrottledError - When the model provider throttles requests

#### Throws

MaxTokensError - When the model reaches its maximum token limit