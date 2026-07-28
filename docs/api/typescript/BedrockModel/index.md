Defined in: [src/models/bedrock.ts:377](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L377)

AWS Bedrock model provider implementation.

Implements the Model interface for AWS Bedrock using the Converse Stream API. Supports streaming responses, tool use, prompt caching, and comprehensive error handling.

## Example

```typescript
const provider = new BedrockModel({
  modelConfig: {
    modelId: 'global.anthropic.claude-sonnet-4-6',
    maxTokens: 1024,
    temperature: 0.7
  },
  clientConfig: {
    region: 'us-west-2'
  }
})

const messages: Message[] = [
  { type: 'message', role: 'user', content: [{ type: 'textBlock', text: 'Hello!' }] }
]

for await (const event of provider.stream(messages)) {
  if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
    process.stdout.write(event.delta.text)
  }
}
```

## Extends

-   [`Model`](/docs/api/typescript/Model/index.md)<[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md)\>

## Constructors

### Constructor

```ts
new BedrockModel(options?): BedrockModel;
```

Defined in: [src/models/bedrock.ts:421](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L421)

Creates a new BedrockModel instance.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options?` | [`BedrockModelOptions`](/docs/api/typescript/BedrockModelOptions/index.md) | Optional configuration for model and client |

#### Returns

`BedrockModel`

#### Example

```typescript
// Minimal configuration with defaults
const provider = new BedrockModel({
  region: 'us-west-2'
})

// With model configuration
const provider = new BedrockModel({
  region: 'us-west-2',
  modelId: 'global.anthropic.claude-sonnet-4-6',
  maxTokens: 2048,
  temperature: 0.8,
  cacheConfig: { strategy: 'auto' }
})

// With client configuration
const provider = new BedrockModel({
  region: 'us-east-1',
  clientConfig: {
    credentials: myCredentials
  }
})
```

#### Overrides

[`Model`](/docs/api/typescript/Model/index.md).[`constructor`](/docs/api/typescript/Model/index.md#constructor)

## Accessors

### modelId

#### Get Signature

```ts
get modelId(): string;
```

Defined in: [src/models/model.ts:227](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/model.ts#L227)

The model ID from the current configuration, if configured.

##### Returns

`string`

#### Inherited from

[`Model`](/docs/api/typescript/Model/index.md).[`modelId`](/docs/api/typescript/Model/index.md#modelid)

---

### stateful

#### Get Signature

```ts
get stateful(): boolean;
```

Defined in: [src/models/model.ts:243](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/model.ts#L243)

Whether this model manages conversation state server-side.

When `true`, the server tracks conversation context across turns, so the SDK sends only the latest message instead of the full history. After each invocation, the agent’s local message history is cleared automatically.

Model providers that support server-side state management should override this to return `true`.

##### Returns

`boolean`

`false` by default

#### Inherited from

[`Model`](/docs/api/typescript/Model/index.md).[`stateful`](/docs/api/typescript/Model/index.md#stateful)

## Methods

### updateConfig()

```ts
updateConfig(modelConfig): void;
```

Defined in: [src/models/bedrock.ts:513](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L513)

Updates the model configuration. Merges the provided configuration with existing settings.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `modelConfig` | [`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md) | Configuration object with model-specific settings to update |

#### Returns

`void`

#### Example

```typescript
// Update temperature and maxTokens
provider.updateConfig({
  temperature: 0.9,
  maxTokens: 2048
})
```

#### Overrides

[`Model`](/docs/api/typescript/Model/index.md).[`updateConfig`](/docs/api/typescript/Model/index.md#updateconfig)

---

### getConfig()

```ts
getConfig(): BedrockModelConfig;
```

Defined in: [src/models/bedrock.ts:528](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L528)

Retrieves the current model configuration.

#### Returns

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md)

The current configuration object

#### Example

```typescript
const config = provider.getConfig()
console.log(config.modelId)
```

#### Overrides

[`Model`](/docs/api/typescript/Model/index.md).[`getConfig`](/docs/api/typescript/Model/index.md#getconfig)

---

### countTokens()

```ts
countTokens(messages, options?): Promise<number>;
```

Defined in: [src/models/bedrock.ts:542](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L542)

Count tokens using Bedrock’s native CountTokens API.

Uses the same message format as the Converse API to get accurate token counts directly from the Bedrock service. Falls back to the base class heuristic on failure.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`Message`](/docs/api/typescript/Message/index.md)\[\] | Array of conversation messages to count tokens for |
| `options?` | [`CountTokensOptions`](/docs/api/typescript/CountTokensOptions/index.md) | Optional options containing system prompt and tool specs |

#### Returns

`Promise`<`number`\>

Total input token count

#### Overrides

[`Model`](/docs/api/typescript/Model/index.md).[`countTokens`](/docs/api/typescript/Model/index.md#counttokens)

---

### stream()

```ts
stream(messages, options?): AsyncIterable<ModelStreamEvent>;
```

Defined in: [src/models/bedrock.ts:623](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/bedrock.ts#L623)

Streams a conversation with the Bedrock model. Returns an async iterable that yields streaming events as they occur.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`Message`](/docs/api/typescript/Message/index.md)\[\] | Array of conversation messages |
| `options?` | [`StreamOptions`](/docs/api/typescript/StreamOptions/index.md) | Optional streaming configuration |

#### Returns

`AsyncIterable`<[`ModelStreamEvent`](/docs/api/typescript/ModelStreamEvent/index.md)\>

Async iterable of streaming events

#### Throws

{ContextWindowOverflowError} When input exceeds the model’s context window

#### Throws

{ModelThrottledError} When Bedrock service throttles requests

#### Example

```typescript
const messages: Message[] = [
  { type: 'message', role: $1, content: [{ type: 'textBlock', text: 'What is 2+2?' }] }
]

const options: StreamOptions = {
  systemPrompt: 'You are a helpful math assistant.',
  toolSpecs: [calculatorTool]
}

for await (const event of provider.stream(messages, options)) {
  if (event.type === 'modelContentBlockDeltaEvent') {
    console.log(event.delta)
  }
}
```

#### Overrides

[`Model`](/docs/api/typescript/Model/index.md).[`stream`](/docs/api/typescript/Model/index.md#stream)

---

### streamAggregated()

```ts
streamAggregated(messages, options?): AsyncGenerator<
  | ContentBlock
| ModelStreamEvent, StreamAggregatedResult, undefined>;
```

Defined in: [src/models/model.ts:329](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/models/model.ts#L329)

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

#### Inherited from

[`Model`](/docs/api/typescript/Model/index.md).[`streamAggregated`](/docs/api/typescript/Model/index.md#streamaggregated)