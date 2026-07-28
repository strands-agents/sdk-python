Defined in: [src/retry/backoff-strategy.ts:16](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/retry/backoff-strategy.ts#L16)

Context passed to a [BackoffStrategy](/docs/api/typescript/BackoffStrategy/index.md) for each retry decision.

Treated as an open, additive-only contract: new optional fields may be added over time, but existing fields will not be removed or repurposed.

## Properties

### attempt

```ts
attempt: number;
```

Defined in: [src/retry/backoff-strategy.ts:18](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/retry/backoff-strategy.ts#L18)

1-based index of the attempt that just failed. Must be >= 1.

---

### elapsedMs

```ts
elapsedMs: number;
```

Defined in: [src/retry/backoff-strategy.ts:20](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/retry/backoff-strategy.ts#L20)

Total milliseconds elapsed since the first attempt started.

---

### lastDelayMs?

```ts
optional lastDelayMs?: number;
```

Defined in: [src/retry/backoff-strategy.ts:22](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/retry/backoff-strategy.ts#L22)

Previously computed delay, if any. Absent before the first retry.