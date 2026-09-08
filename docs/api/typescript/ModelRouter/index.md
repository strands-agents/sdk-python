Defined in: [src/models/routing/router.ts:129](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L129)

Routes each agent invocation among an immutable set of candidate models.

The default [FallbackStrategy](/docs/api/typescript/FallbackStrategy/index.md) prefers the candidate with the fewest recorded failures and breaks ties by declaration order. `maxSwitches` bounds successful candidate changes per invocation.

## Example

```typescript
const router = new ModelRouter([
  new RoutingCandidate({ model: primary, name: 'primary' }),
  new RoutingCandidate({ model: fallback, name: 'fallback' }),
])
const agent = new Agent({ model: router })
```

## Implements

-   [`Plugin`](/docs/api/typescript/Plugin/index.md)

## Constructors

### Constructor

```ts
new ModelRouter(models, options?): ModelRouter;
```

Defined in: [src/models/routing/router.ts:145](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L145)

Create a model router.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `models` | readonly [`CandidateInput`](/docs/api/typescript/CandidateInput/index.md)\[\] | Candidate models, nested routers, or candidate wrappers |
| `options` | [`ModelRouterOptions`](/docs/api/typescript/ModelRouterOptions/index.md) | Routing strategy and switch cap |

#### Returns

`ModelRouter`

#### Throws

TypeError if models or the strategy are invalid

#### Throws

Error if candidates are empty, duplicated, named alike, stateful, or `maxSwitches` is not a non-negative integer

## Properties

### name

```ts
readonly name: "strands:model-router" = 'strands:model-router';
```

Defined in: [src/models/routing/router.ts:130](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L130)

A stable string identifier for the plugin. Used for logging, duplicate detection, and plugin management.

For strands-vended plugins, names should be prefixed with `strands:`.

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`name`](/docs/api/typescript/Plugin/index.md#name)

## Accessors

### candidates

#### Get Signature

```ts
get candidates(): readonly RoutingCandidate[];
```

Defined in: [src/models/routing/router.ts:164](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L164)

Normalized candidates in declaration order.

##### Returns

readonly [`RoutingCandidate`](/docs/api/typescript/RoutingCandidate/index.md)\[\]

---

### defaultModel

#### Get Signature

```ts
get defaultModel(): Model;
```

Defined in: [src/models/routing/router.ts:169](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L169)

First declared candidate resolved without consulting a strategy.

##### Returns

[`Model`](/docs/api/typescript/Model/index.md)

## Methods

### initAgent()

```ts
initAgent(agent): void;
```

Defined in: [src/models/routing/router.ts:191](https://github.com/strands-agents/harness-sdk/blob/a642b4177ba14dba41ef296d510d68f718f60ebb/strands-ts/src/models/routing/router.ts#L191)

Register routing middleware and lifecycle hooks.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `agent` | `LocalAgent` | Agent using this router as its model |

#### Returns

`void`

#### Throws

Error if attached as an ordinary plugin rather than as the model

#### Implementation of

[`Plugin`](/docs/api/typescript/Plugin/index.md).[`initAgent`](/docs/api/typescript/Plugin/index.md#initagent)