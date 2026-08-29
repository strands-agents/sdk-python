Defined in: [src/models/routing/router.ts:51](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/models/routing/router.ts#L51)

A model or opaque model group with an optional name and description.

Base instances are frozen automatically. Subclasses must freeze themselves after initializing additional fields.

## Constructors

### Constructor

```ts
new RoutingCandidate(options): RoutingCandidate;
```

Defined in: [src/models/routing/router.ts:64](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/models/routing/router.ts#L64)

Create an immutable routing candidate.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`RoutingCandidateOptions`](/docs/api/typescript/RoutingCandidateOptions/index.md) | Candidate model, name, and description |

#### Returns

`RoutingCandidate`

## Properties

### model

```ts
readonly model:
  | Model<BaseModelConfig>
  | ModelRouter;
```

Defined in: [src/models/routing/router.ts:53](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/models/routing/router.ts#L53)

Concrete model or opaque nested router.

---

### name?

```ts
readonly optional name?: string;
```

Defined in: [src/models/routing/router.ts:55](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/models/routing/router.ts#L55)

Optional strategy-facing name.

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/models/routing/router.ts:57](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/models/routing/router.ts#L57)

Optional strategy-facing description.