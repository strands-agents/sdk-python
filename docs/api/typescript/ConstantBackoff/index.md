Defined in: [src/retry/backoff-strategy.ts:68](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L68)

Constant backoff: returns the same delay for every retry.

## Implements

-   [`BackoffStrategy`](/docs/api/typescript/BackoffStrategy/index.md)

## Constructors

### Constructor

```ts
new ConstantBackoff(opts?): ConstantBackoff;
```

Defined in: [src/retry/backoff-strategy.ts:71](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L71)

#### Parameters

| Parameter | Type |
| --- | --- |
| `opts` | [`ConstantBackoffOptions`](/docs/api/typescript/ConstantBackoffOptions/index.md) |

#### Returns

`ConstantBackoff`

## Methods

### nextDelay()

```ts
nextDelay(ctx): number;
```

Defined in: [src/retry/backoff-strategy.ts:75](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/retry/backoff-strategy.ts#L75)

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