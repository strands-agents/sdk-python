Defined in: [src/retry/backoff-strategy.ts:132](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/retry/backoff-strategy.ts#L132)

Exponential backoff: delay grows as `baseMs * multiplier^(attempt-1)`, capped at `maxMs`, then jittered.

## Implements

-   [`BackoffStrategy`](/docs/api/typescript/BackoffStrategy/index.md)

## Constructors

### Constructor

```ts
new ExponentialBackoff(opts?): ExponentialBackoff;
```

Defined in: [src/retry/backoff-strategy.ts:138](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/retry/backoff-strategy.ts#L138)

#### Parameters

| Parameter | Type |
| --- | --- |
| `opts` | [`ExponentialBackoffOptions`](/docs/api/typescript/ExponentialBackoffOptions/index.md) |

#### Returns

`ExponentialBackoff`

## Methods

### nextDelay()

```ts
nextDelay(ctx): number;
```

Defined in: [src/retry/backoff-strategy.ts:145](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/retry/backoff-strategy.ts#L145)

Returns the delay in milliseconds before the next attempt.

Must be a non-negative finite number. Implementations should treat `ctx.attempt < 1` as a programmer error.

#### Parameters

| Parameter | Type |
| --- | --- |
| `ctx` | [`BackoffContext`](/docs/api/typescript/BackoffContext/index.md) |

#### Returns

`number`

#### Implementation of

[`BackoffStrategy`](/docs/api/typescript/BackoffStrategy/index.md).[`nextDelay`](/docs/api/typescript/BackoffStrategy/index.md#nextdelay)