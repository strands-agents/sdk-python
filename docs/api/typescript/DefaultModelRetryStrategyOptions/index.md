Defined in: [src/retry/default-model-retry-strategy.ts:29](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/default-model-retry-strategy.ts#L29)

Options for [DefaultModelRetryStrategy](/docs/api/typescript/DefaultModelRetryStrategy/index.md).

## Properties

### maxAttempts?

```ts
optional maxAttempts?: number;
```

Defined in: [src/retry/default-model-retry-strategy.ts:34](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/default-model-retry-strategy.ts#L34)

Total model attempts before giving up and re-raising the error. Must be >= 1. Default DEFAULT\_MAX\_ATTEMPTS.

---

### backoff?

```ts
optional backoff?: BackoffStrategy;
```

Defined in: [src/retry/default-model-retry-strategy.ts:39](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/retry/default-model-retry-strategy.ts#L39)

Backoff used to compute the delay between retries. Default: `new ExponentialBackoff({ baseMs: DEFAULT_BACKOFF_BASE_MS, maxMs: DEFAULT_BACKOFF_MAX_MS })`.