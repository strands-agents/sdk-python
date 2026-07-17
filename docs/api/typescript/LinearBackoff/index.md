Defined in: [src/retry/backoff-strategy.ts:96](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/retry/backoff-strategy.ts#L96)

Linear backoff: delay grows as `baseMs * attempt`, capped at `maxMs`, then jittered.

## Implements

-   [`BackoffStrategy`](/docs/api/typescript/BackoffStrategy/index.md)

## Constructors

### Constructor

```ts
new LinearBackoff(opts?): LinearBackoff;
```

Defined in: [src/retry/backoff-strategy.ts:101](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/retry/backoff-strategy.ts#L101)

#### Parameters

| Parameter | Type |
| --- | --- |
| `opts` | [`LinearBackoffOptions`](/docs/api/typescript/LinearBackoffOptions/index.md) |

#### Returns

`LinearBackoff`

## Methods

### nextDelay()

```ts
nextDelay(ctx): number;
```

Defined in: [src/retry/backoff-strategy.ts:107](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/retry/backoff-strategy.ts#L107)

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