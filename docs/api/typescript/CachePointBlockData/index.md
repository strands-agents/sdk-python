Defined in: [src/types/messages.ts:573](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L573)

Data for a cache point block.

## Properties

### cacheType

```ts
cacheType: "default";
```

Defined in: [src/types/messages.ts:577](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L577)

The cache type. Currently only ‘default’ is supported.

---

### ttl?

```ts
optional ttl?: string;
```

Defined in: [src/types/messages.ts:586](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L586)

Optional TTL for the cache entry. When omitted, the provider’s default TTL is used.

The accepted value space is provider-specific. For example, the Bedrock provider only accepts the values defined by `BedrockCacheTTL` (`'5m'` and `'1h'`). Other providers may accept different values or ignore this field.