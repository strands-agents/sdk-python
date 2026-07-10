```ts
type JitterKind = "none" | "full" | "equal" | "decorrelated";
```

Defined in: [src/retry/backoff-strategy.ts:49](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/retry/backoff-strategy.ts#L49)

Supported jitter modes.

-   `none`: return the raw delay unchanged
-   `full`: uniform random in `[0, raw]`
-   `equal`: `raw/2 + uniform(0, raw/2)` (half fixed, half random)
-   `decorrelated`: `uniform(baseMs, lastDelayMs * 3)`, capped at `maxMs`; falls back to `full` on the first retry when `lastDelayMs` is unavailable

For jitter outside these modes, implement [BackoffStrategy](/docs/api/typescript/BackoffStrategy/index.md) directly.