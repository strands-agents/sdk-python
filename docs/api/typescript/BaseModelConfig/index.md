Defined in: [src/models/model.ts:170](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L170)

Base configuration interface for all model providers.

This interface defines the common configuration properties that all model providers should support. Provider-specific configurations should extend this interface.

## Extended by

-   [`BedrockModelConfig`](/docs/api/typescript/BedrockModelConfig/index.md)

## Properties

### modelId?

```ts
optional modelId?: string;
```

Defined in: [src/models/model.ts:175](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L175)

The model identifier. This typically specifies which model to use from the provider’s catalog.

---

### maxTokens?

```ts
optional maxTokens?: number;
```

Defined in: [src/models/model.ts:182](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L182)

Maximum number of tokens to generate in the response.

#### See

Provider-specific documentation for exact behavior

---

### temperature?

```ts
optional temperature?: number;
```

Defined in: [src/models/model.ts:189](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L189)

Controls randomness in generation.

#### See

Provider-specific documentation for valid range

---

### topP?

```ts
optional topP?: number;
```

Defined in: [src/models/model.ts:196](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L196)

Controls diversity via nucleus sampling.

#### See

Provider-specific documentation for details

---

### contextWindowLimit?

```ts
optional contextWindowLimit?: number;
```

Defined in: [src/models/model.ts:208](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/model.ts#L208)

Maximum context window size in tokens for the model.

This value represents the total token capacity shared between input and output. When not provided, it is automatically resolved from a built-in lookup table based on the configured model ID. An explicit value always takes precedence.

When `modelId` is changed via `updateConfig()`, this value is automatically re-resolved if it was initially auto-populated. Explicitly set values are preserved.