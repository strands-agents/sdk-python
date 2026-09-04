```ts
type BedrockCacheTTL = CacheTTL;
```

Defined in: [src/models/bedrock.ts:147](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/bedrock.ts#L147)

TTL durations accepted by Bedrock for prompt-cache checkpoints.

Bedrock accepts `'5m'` (default) and `'1h'`, and validates the value server-side, rejecting unsupported values with `ValidationException`.

Bedrock also requires checkpoint TTLs to be **non-increasing** across `toolConfig` → system → messages — setting a longer TTL on a later checkpoint than an earlier one will be rejected by the service.

## See

[https://docs.aws.amazon.com/bedrock/latest/APIReference/API\_runtime\_CachePointBlock.html](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html)