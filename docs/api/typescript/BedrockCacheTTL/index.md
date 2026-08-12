```ts
type BedrockCacheTTL = CacheTTL;
```

Defined in: [src/models/bedrock.ts:145](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/bedrock.ts#L145)

TTL durations accepted by Bedrock for prompt-cache checkpoints.

Bedrock accepts `'5m'` (default) and `'1h'`, and validates the value server-side, rejecting unsupported values with `ValidationException`.

Bedrock also requires checkpoint TTLs to be **non-increasing** across `toolConfig` → system → messages — setting a longer TTL on a later checkpoint than an earlier one will be rejected by the service.

## See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_CachePointBlock.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html)