Defined in: [src/models/model.ts:72](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L72)

Configuration for prompt caching.

Providers consume only the fields they support.

## Properties

### strategy?

```ts
optional strategy?: "auto" | "anthropic";
```

Defined in: [src/models/model.ts:81](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L81)

Whether to skip caching for models that do not support it.

-   “auto”: cache only when the model is known to support it
-   “anthropic”: cache without that check, for model identifiers it cannot inspect (an application inference profile, for example)

#### Default Value

```ts
'auto'
```

---

### ttl?

```ts
optional ttl?: CacheTTL;
```

Defined in: [src/models/model.ts:93](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L93)

TTL for every cache point, overridden by a per-section TTL. Provider default when omitted.

Bedrock requires checkpoint TTLs to be non-increasing across `toolConfig`, system and messages, and rejects a longer TTL that follows a shorter one. This TTL therefore also fills in for a cache point placed by hand in the system prompt that carries none of its own, so one value keeps every checkpoint in step. A TTL written on such a point is left as written, and a `toolsTTL` that differs from this one leaves the point at the provider default rather than landing a longer TTL behind a shorter checkpoint - either way, two TTLs in tension are yours to reconcile.

---

### toolsTTL?

```ts
optional toolsTTL?: boolean | CacheTTL;
```

Defined in: [src/models/model.ts:100](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L100)

Cache the tool definitions. A TTL sets this section’s duration; `false` disables it.

#### Default Value

```ts
true
```

---

### systemPromptTTL?

```ts
optional systemPromptTTL?: boolean | CacheTTL;
```

Defined in: [src/models/model.ts:109](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L109)

Cache the system prompt, auto-injecting a cache point at its end so repeated calls with the same static system prefix hit the cache. A TTL sets this section’s duration; `true` (the default) reads the value from `ttl`; `false` disables systemPrompt cache injection.

#### Default Value

```ts
true
```

---

### messagesTTL?

```ts
optional messagesTTL?: boolean | CacheTTL;
```

Defined in: [src/models/model.ts:117](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L117)

Cache the conversation prefix, on the last user message. A TTL sets this section’s duration; `false` disables it.

#### Default Value

```ts
true
```

---

### cacheKey?

```ts
optional cacheKey?: string;
```

Defined in: [src/models/model.ts:120](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/model.ts#L120)

Stable identity a provider can use to route its cache.