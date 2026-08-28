Defined in: [src/memory/types.ts:200](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/memory/types.ts#L200)

Configuration for the `add_memory` tool. Extends [MemoryToolConfig](/docs/api/typescript/MemoryToolConfig/index.md) with an explicit allowlist of stores the tool may write to.

## Extends

-   [`MemoryToolConfig`](/docs/api/typescript/MemoryToolConfig/index.md)

## Properties

### name?

```ts
optional name?: string;
```

Defined in: [src/memory/types.ts:191](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/memory/types.ts#L191)

Custom tool name.

#### Inherited from

[`MemoryToolConfig`](/docs/api/typescript/MemoryToolConfig/index.md).[`name`](/docs/api/typescript/MemoryToolConfig/index.md#name)

---

### description?

```ts
optional description?: string;
```

Defined in: [src/memory/types.ts:193](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/memory/types.ts#L193)

Custom tool description.

#### Inherited from

[`MemoryToolConfig`](/docs/api/typescript/MemoryToolConfig/index.md).[`description`](/docs/api/typescript/MemoryToolConfig/index.md#description)

---

### stores?

```ts
optional stores?: (string | MemoryStore)[];
```

Defined in: [src/memory/types.ts:206](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/memory/types.ts#L206)

The writable stores the `add_memory` tool may write to, given as store names or the [MemoryStore](/docs/api/typescript/MemoryStore/index.md) instances themselves. Each must be a configured, `writable` store. Omit (or set `addToolConfig: true`) to allow all writable stores.

---

### waitForWrites?

```ts
optional waitForWrites?: boolean;
```

Defined in: [src/memory/types.ts:214](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/memory/types.ts#L214)

Whether the tool waits for store writes before returning to the model. Defaults to `true`.

-   `true` (default): waits for writes — the tool returns `{ stored }` on success, or surfaces a failure to the model if any store write fails.
-   `false`: fire-and-forget — the tool returns `{ accepted }` once writes are dispatched (so a slow backend never blocks the agent loop); per-store failures are logged.