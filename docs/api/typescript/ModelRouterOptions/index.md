Defined in: [src/models/routing/router.ts:85](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/router.ts#L85)

Options for constructing a [ModelRouter](/docs/api/typescript/ModelRouter/index.md).

## Properties

### strategy?

```ts
readonly optional strategy?: RoutingStrategy;
```

Defined in: [src/models/routing/router.ts:87](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/router.ts#L87)

Strategy responsible for every routing decision.

---

### maxSwitches?

```ts
readonly optional maxSwitches?: number;
```

Defined in: [src/models/routing/router.ts:89](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/routing/router.ts#L89)

Maximum successful candidate switches per invocation.