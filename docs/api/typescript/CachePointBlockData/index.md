Defined in: [src/types/messages.ts:575](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/types/messages.ts#L575)

Data for a cache point block.

## Properties

### cacheType

```ts
cacheType: "default";
```

Defined in: [src/types/messages.ts:579](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/types/messages.ts#L579)

The cache type. Currently only ‘default’ is supported.

---

### ttl?

```ts
optional ttl?: string;
```

Defined in: [src/types/messages.ts:588](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/types/messages.ts#L588)

Optional TTL for the cache entry. When omitted, the provider’s default TTL is used.

The accepted value space is provider-specific. For example, the Bedrock provider only accepts the values defined by `BedrockCacheTTL` (`'5m'` and `'1h'`). Other providers may accept different values or ignore this field.