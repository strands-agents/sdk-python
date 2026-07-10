Defined in: [src/models/bedrock.ts:150](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/bedrock.ts#L150)

Bedrock-specific prompt-caching configuration. Narrows the TTL fields onto the common [CacheConfig](/docs/api/typescript/CacheConfig/index.md) for the Bedrock provider.

## Extends

-   [`CacheConfig`](/docs/api/typescript/CacheConfig/index.md)

## Properties

### toolsTTL?

```ts
optional toolsTTL?: BedrockCacheTTL;
```

Defined in: [src/models/bedrock.ts:152](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/bedrock.ts#L152)

TTL applied to the auto-injected cache point appended after `toolConfig.tools`.

---

### messagesTTL?

```ts
optional messagesTTL?: BedrockCacheTTL;
```

Defined in: [src/models/bedrock.ts:155](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/bedrock.ts#L155)

TTL applied to the auto-injected cache point appended to the last user message.

---

### strategy

```ts
strategy: "auto" | "anthropic";
```

Defined in: [src/models/model.ts:76](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/model.ts#L76)

Caching strategy to use.

-   “auto”: Automatically inject cache points at optimal positions based on model ID detection (after tools, after last user message)
-   “anthropic”: Force enable Anthropic-style caching (useful for application inference profiles)

#### Inherited from

[`CacheConfig`](/docs/api/typescript/CacheConfig/index.md).[`strategy`](/docs/api/typescript/CacheConfig/index.md#strategy)