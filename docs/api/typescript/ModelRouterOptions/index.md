Defined in: [src/models/routing/router.ts:85](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/router.ts#L85)

Options for constructing a [ModelRouter](/docs/api/typescript/ModelRouter/index.md).

## Properties

### strategy?

```ts
readonly optional strategy?: RoutingStrategy;
```

Defined in: [src/models/routing/router.ts:87](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/router.ts#L87)

Strategy responsible for every routing decision.

---

### maxSwitches?

```ts
readonly optional maxSwitches?: number;
```

Defined in: [src/models/routing/router.ts:89](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/models/routing/router.ts#L89)

Maximum successful candidate switches per invocation.