Defined in: [src/models/bedrock.ts:328](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L328)

Options for creating a BedrockModel instance.

## Extends

-   [`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md)

## Properties

### maxTokens?

```ts
optional maxTokens?: number;
```

Defined in: [src/models/bedrock.ts:245](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L245)

Maximum number of tokens to generate in the response.

#### See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_InferenceConfiguration.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`maxTokens`](/docs/api/typescript/BedrockModelConfig/index.md#maxtokens)

---

### temperature?

```ts
optional temperature?: number;
```

Defined in: [src/models/bedrock.ts:252](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L252)

Controls randomness in generation.

#### See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_InferenceConfiguration.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`temperature`](/docs/api/typescript/BedrockModelConfig/index.md#temperature)

---

### topP?

```ts
optional topP?: number;
```

Defined in: [src/models/bedrock.ts:259](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L259)

Controls diversity via nucleus sampling.

#### See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_InferenceConfiguration.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`topP`](/docs/api/typescript/BedrockModelConfig/index.md#topp)

---

### stopSequences?

```ts
optional stopSequences?: string[];
```

Defined in: [src/models/bedrock.ts:264](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L264)

Array of sequences that will stop generation when encountered.

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`stopSequences`](/docs/api/typescript/BedrockModelConfig/index.md#stopsequences)

---

### cacheConfig?

```ts
optional cacheConfig?: BedrockCacheConfig;
```

Defined in: [src/models/bedrock.ts:271](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L271)

Configuration for prompt caching.

#### See

[https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`cacheConfig`](/docs/api/typescript/BedrockModelConfig/index.md#cacheconfig)

---

### additionalRequestFields?

```ts
optional additionalRequestFields?: JSONValue;
```

Defined in: [src/models/bedrock.ts:276](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L276)

Additional fields to include in the Bedrock request.

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`additionalRequestFields`](/docs/api/typescript/BedrockModelConfig/index.md#additionalrequestfields)

---

### additionalResponseFieldPaths?

```ts
optional additionalResponseFieldPaths?: string[];
```

Defined in: [src/models/bedrock.ts:281](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L281)

Additional response field paths to extract from the Bedrock response.

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`additionalResponseFieldPaths`](/docs/api/typescript/BedrockModelConfig/index.md#additionalresponsefieldpaths)

---

### additionalArgs?

```ts
optional additionalArgs?: JSONValue;
```

Defined in: [src/models/bedrock.ts:287](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L287)

Additional arguments to pass through to the Bedrock Converse API.

#### See

[https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/client/bedrock-runtime/command/ConverseStreamCommand/](https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/client/bedrock-runtime/command/ConverseStreamCommand/)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`additionalArgs`](/docs/api/typescript/BedrockModelConfig/index.md#additionalargs)

---

### stream?

```ts
optional stream?: boolean;
```

Defined in: [src/models/bedrock.ts:297](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L297)

Whether or not to stream responses from the model.

This will use the ConverseStream API instead of the Converse API.

#### See

-   [https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_Converse.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
-   [https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_ConverseStream.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`stream`](/docs/api/typescript/BedrockModelConfig/index.md#stream)

---

### includeToolResultStatus?

```ts
optional includeToolResultStatus?: boolean | "auto";
```

Defined in: [src/models/bedrock.ts:305](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L305)

Flag to include status field in tool results.

-   `true`: Always include status field
-   `false`: Never include status field
-   `'auto'`: Automatically determine based on model ID (default)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`includeToolResultStatus`](/docs/api/typescript/BedrockModelConfig/index.md#includetoolresultstatus)

---

### guardrailConfig?

```ts
optional guardrailConfig?: BedrockGuardrailConfig;
```

Defined in: [src/models/bedrock.ts:311](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L311)

Guardrail configuration for content filtering and safety controls.

#### See

[https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`guardrailConfig`](/docs/api/typescript/BedrockModelConfig/index.md#guardrailconfig)

---

### useNativeTokenCount?

```ts
optional useNativeTokenCount?: boolean;
```

Defined in: [src/models/bedrock.ts:322](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L322)

Whether to use the native Bedrock CountTokens API.

When `true`, `countTokens()` calls the Bedrock CountTokens API for accurate counts. When `false` or not set (default), skips the API call and uses the character-based heuristic estimator.

#### Default Value

```ts
false
```

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`useNativeTokenCount`](/docs/api/typescript/BedrockModelConfig/index.md#usenativetokencount)

---

### region?

```ts
optional region?: string;
```

Defined in: [src/models/bedrock.ts:332](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L332)

AWS region to use for the Bedrock service.

---

### clientConfig?

```ts
optional clientConfig?: BedrockRuntimeClientConfig;
```

Defined in: [src/models/bedrock.ts:337](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L337)

Configuration for the Bedrock Runtime client.

---

### apiKey?

```ts
optional apiKey?: string;
```

Defined in: [src/models/bedrock.ts:344](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/bedrock.ts#L344)

Amazon Bedrock API key for bearer token authentication. When provided, requests use the API key instead of SigV4 signing.

#### See

[https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)

---

### modelId?

```ts
optional modelId?: string;
```

Defined in: [src/models/model.ts:91](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/model.ts#L91)

The model identifier. This typically specifies which model to use from the provider’s catalog.

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`modelId`](/docs/api/typescript/BedrockModelConfig/index.md#modelid)

---

### contextWindowLimit?

```ts
optional contextWindowLimit?: number;
```

Defined in: [src/models/model.ts:124](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/models/model.ts#L124)

Maximum context window size in tokens for the model.

This value represents the total token capacity shared between input and output. When not provided, it is automatically resolved from a built-in lookup table based on the configured model ID. An explicit value always takes precedence.

When `modelId` is changed via `updateConfig()`, this value is automatically re-resolved if it was initially auto-populated. Explicitly set values are preserved.

#### Inherited from

[`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md).[`contextWindowLimit`](/docs/api/typescript/BedrockModelConfig/index.md#contextwindowlimit)