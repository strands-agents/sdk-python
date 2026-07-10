```ts
type BedrockCacheTTL =
  | "5m"
  | "1h"
  | string & {
};
```

Defined in: [src/models/bedrock.ts:144](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/bedrock.ts#L144)

TTL durations accepted by Bedrock for prompt-cache checkpoints.

Bedrock currently accepts `'5m'` (default) and `'1h'`. The `(string & {})` branch keeps autocomplete on the known values while letting callers pass any string forward — Bedrock validates the value server-side and rejects unsupported values with `ValidationException`, so this stays correct as AWS adds new TTL values without an SDK update.

Bedrock also requires checkpoint TTLs to be **non-increasing** across `toolConfig` → system → messages — setting a longer TTL on a later checkpoint than an earlier one will be rejected by the service.

## See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_CachePointBlock.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html)