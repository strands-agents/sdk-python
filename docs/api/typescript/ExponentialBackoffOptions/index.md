Defined in: [src/retry/backoff-strategy.ts:117](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L117)

Options for [ExponentialBackoff](/docs/api/typescript/ExponentialBackoff/index.md).

## Properties

### baseMs?

```ts
optional baseMs?: number;
```

Defined in: [src/retry/backoff-strategy.ts:119](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L119)

Base delay in ms. Delay grows as `baseMs * multiplier^(attempt-1)`. Default 1000.

---

### maxMs?

```ts
optional maxMs?: number;
```

Defined in: [src/retry/backoff-strategy.ts:121](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L121)

Upper bound applied before jitter. Default 30\_000.

---

### multiplier?

```ts
optional multiplier?: number;
```

Defined in: [src/retry/backoff-strategy.ts:123](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L123)

Growth factor per attempt. Default 2.

---

### jitter?

```ts
optional jitter?: JitterKind;
```

Defined in: [src/retry/backoff-strategy.ts:125](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L125)

Jitter mode. Default ‘full’.