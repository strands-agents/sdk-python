Defined in: [src/models/routing/fallback-strategy.ts:5](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/fallback-strategy.ts#L5)

Selects the healthiest candidate not yet tried since the last success.

## Implements

-   [`RoutingStrategy`](/docs/api/typescript/RoutingStrategy/index.md)

## Constructors

### Constructor

```ts
new FallbackStrategy(): FallbackStrategy;
```

#### Returns

`FallbackStrategy`

## Methods

### select()

```ts
select(context): Promise<RoutingCandidate>;
```

Defined in: [src/models/routing/fallback-strategy.ts:12](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/fallback-strategy.ts#L12)

Select the least-failed available candidate, breaking ties by declaration order.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`RoutingContext`](/docs/api/typescript/RoutingContext/index.md) | Current routing context |

#### Returns

`Promise`<[`RoutingCandidate`](/docs/api/typescript/RoutingCandidate/index.md)\>

The selected candidate, or `undefined` when the round is exhausted

#### Implementation of

[`RoutingStrategy`](/docs/api/typescript/RoutingStrategy/index.md).[`select`](/docs/api/typescript/RoutingStrategy/index.md#select)