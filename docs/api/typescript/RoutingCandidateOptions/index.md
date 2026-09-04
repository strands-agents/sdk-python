Defined in: [src/models/routing/router.ts:37](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L37)

Construction data for a [RoutingCandidate](/docs/api/typescript/RoutingCandidate/index.md).

## Properties

### model

```ts
readonly model:
  | Model<BaseModelConfig>
  | ModelRouter;
```

Defined in: [src/models/routing/router.ts:39](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L39)

Concrete model or opaque nested router.

---

### name?

```ts
readonly optional name?: string;
```

Defined in: [src/models/routing/router.ts:41](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L41)

Optional strategy-facing name.

---

### description?

```ts
readonly optional description?: string;
```

Defined in: [src/models/routing/router.ts:43](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L43)

Optional strategy-facing description.

---

### metadata?

```ts
readonly optional metadata?: Readonly<Record<string, JSONValue>>;
```

Defined in: [src/models/routing/router.ts:45](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/routing/router.ts#L45)

Optional strategy-facing evidence; must be JSON-serializable and free of secrets.