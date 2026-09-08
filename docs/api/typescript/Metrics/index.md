Defined in: [src/models/streaming.ts:525](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/streaming.ts#L525)

Performance metrics for a model invocation.

## Properties

### latencyMs

```ts
latencyMs: number;
```

Defined in: [src/models/streaming.ts:529](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/streaming.ts#L529)

Latency in milliseconds.

---

### timeToFirstByteMs?

```ts
optional timeToFirstByteMs?: number;
```

Defined in: [src/models/streaming.ts:535](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/streaming.ts#L535)

Time to first byte in milliseconds. Latency from sending the model request to receiving the first content chunk.