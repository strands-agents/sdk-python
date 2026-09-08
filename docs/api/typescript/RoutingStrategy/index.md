Defined in: [src/models/routing/strategy.ts:32](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L32)

Chooses a configured routing candidate.

## Methods

### select()

```ts
select(context): Promise<RoutingCandidate>;
```

Defined in: [src/models/routing/strategy.ts:52](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/strategy.ts#L52)

Select a candidate from `context.candidates`, or decline with `undefined`.

The router asks before the first model call with an empty attempt history, then after unclaimed failures until routing stops. The returned candidate must be the same instance as one in `context.candidates`.

Declining the opening selection uses the router’s default model. Declining after a failure ends routing and preserves the pending model error. During opening selection, strategy errors, invalid candidates, and candidate-resolution errors propagate. After a model failure, strategy errors and invalid candidates end routing without replacing the pending error.

Each failure round may use a candidate once. Returning a candidate already used in the current round ends routing. If a nested candidate cannot resolve after a failure, it consumes its round slot and the strategy is asked again.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`RoutingContext`](/docs/api/typescript/RoutingContext/index.md) | Current request and chronological routing history |

#### Returns

`Promise`<[`RoutingCandidate`](/docs/api/typescript/RoutingCandidate/index.md)\>

A configured candidate, or `undefined` to decline