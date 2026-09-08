Defined in: [src/retry/default-model-retry-strategy.ts:24](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/retry/default-model-retry-strategy.ts#L24)

Options for [DefaultModelRetryStrategy](/docs/api/typescript/DefaultModelRetryStrategy/index.md).

## Properties

### maxAttempts?

```ts
optional maxAttempts?: number;
```

Defined in: [src/retry/default-model-retry-strategy.ts:29](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/retry/default-model-retry-strategy.ts#L29)

Total model attempts before giving up and re-raising the error. Must be >= 1. Default DEFAULT\_MAX\_ATTEMPTS.

---

### backoff?

```ts
optional backoff?: BackoffStrategy;
```

Defined in: [src/retry/default-model-retry-strategy.ts:34](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/retry/default-model-retry-strategy.ts#L34)

Backoff used to compute the delay between retries. Default: `new ExponentialBackoff({ baseMs: DEFAULT_BACKOFF_BASE_MS, maxMs: DEFAULT_BACKOFF_MAX_MS })`.