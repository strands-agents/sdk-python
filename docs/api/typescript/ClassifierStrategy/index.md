Defined in: [src/models/routing/classifier-strategy.ts:105](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/classifier-strategy.ts#L105)

Choose a candidate by applying a configurable policy with a classifier model.

Classification adds one call to the explicitly configured model. Candidate declaration order does not inform classification. Candidate names, descriptions, metadata, the latest request, and textual parent-agent instructions may cross the classifier provider boundary and must not contain secrets. Structured parent-system-prompt blocks such as cache points are omitted because the classifier receives rebuilt, bounded context rather than the original prompt.

Classification failures warn and decline selection, so [ModelRouter](/docs/api/typescript/ModelRouter/index.md) serves candidate zero. If the selected candidate later fails, this strategy declines further selection and lets the original model error surface without switching. Nested routers are treated as opaque candidates using only their wrapper evidence.

## Example

```typescript
const router = new ModelRouter(
  [
    new RoutingCandidate({ model: fast, name: 'routine', metadata: { supportsToolUse: true } }),
    new RoutingCandidate({ model: strong, name: 'complex' }),
  ],
  { strategy: new ClassifierStrategy(classifierModel) }
)
const agent = new Agent({ model: router })
```

## Implements

-   [`RoutingStrategy`](/docs/api/typescript/RoutingStrategy/index.md)

## Constructors

### Constructor

```ts
new ClassifierStrategy(model, options?): ClassifierStrategy;
```

Defined in: [src/models/routing/classifier-strategy.ts:121](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/classifier-strategy.ts#L121)

Create a classifier strategy.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | [`Model`](/docs/api/typescript/Model/index.md) | Model used for classification; it must honor forced tool selection (`toolChoice`), since a provider that ignores it fails classification silently and every selection falls back to candidate zero |
| `options` | [`ClassifierStrategyOptions`](/docs/api/typescript/ClassifierStrategyOptions/index.md) | Routing policy, timeout, and character budgets |

#### Returns

`ClassifierStrategy`

#### Throws

TypeError if `model` is not a Model

#### Throws

Error if `timeoutMs` is not finite and greater than zero or a character limit is not a positive integer

## Methods

### select()

```ts
select(context): Promise<RoutingCandidate>;
```

Defined in: [src/models/routing/classifier-strategy.ts:152](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/classifier-strategy.ts#L152)

Select one opening candidate, declining on classification or serving-time failure.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`RoutingContext`](/docs/api/typescript/RoutingContext/index.md) | Current request and chronological routing history |

#### Returns

`Promise`<[`RoutingCandidate`](/docs/api/typescript/RoutingCandidate/index.md)\>

The classified candidate, or `undefined` to decline

#### Throws

Error if the candidates’ serialized evidence exceeds `maxCandidateChars`; this misconfiguration is permanent, so it propagates instead of declining

#### Implementation of

[`RoutingStrategy`](/docs/api/typescript/RoutingStrategy/index.md).[`select`](/docs/api/typescript/RoutingStrategy/index.md#select)