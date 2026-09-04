Defined in: [src/models/routing/router.ts:57](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L57)

A model or opaque model group with optional strategy-facing evidence.

Classifier-based strategies may send `name`, `description`, and `metadata` across provider boundaries, so they must not contain secrets. Metadata is stored without copying, so it must not be mutated after construction.

Base instances are frozen automatically. Subclasses must freeze themselves after initializing additional fields.

## Constructors

### Constructor

```ts
new RoutingCandidate(options): RoutingCandidate;
```

Defined in: [src/models/routing/router.ts:75](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L75)

Create an immutable routing candidate.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `options` | [`RoutingCandidateOptions`](/docs/api/typescript/RoutingCandidateOptions/index.md) | Candidate model, name, description, and metadata |

#### Returns

`RoutingCandidate`

#### Throws

TypeError if metadata is not a plain object

#### Throws

JsonValidationError if metadata contains values that cannot be serialized to JSON

#### Throws

Error if metadata serialization fails for another reason

## Properties

### model

```ts
readonly model:
  | Model<BaseModelConfig>
  | ModelRouter;
```

Defined in: [src/models/routing/router.ts:59](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L59)

Concrete model or opaque nested router.

---

### name?

```ts
readonly optional name?: string;
```

Defined in: [src/models/routing/router.ts:61](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L61)

Optional strategy-facing name.

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/models/routing/router.ts:63](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L63)

Optional strategy-facing description.

---

### metadata?

```ts
readonly optional metadata?: Readonly<Record<string, JSONValue>>;
```

Defined in: [src/models/routing/router.ts:65](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L65)

Optional strategy-facing evidence; must be JSON-serializable and free of secrets.