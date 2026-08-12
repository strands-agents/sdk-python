Defined in: [src/memory/types.ts:265](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/types.ts#L265)

Configuration for the [MemoryManager](/docs/api/typescript/MemoryManager/index.md).

## Properties

### stores

```ts
stores: MemoryStore[];
```

Defined in: [src/memory/types.ts:267](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/types.ts#L267)

One or more memory stores to manage.

---

### searchToolConfig?

```ts
optional searchToolConfig?: boolean | MemoryToolConfig;
```

Defined in: [src/memory/types.ts:269](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/types.ts#L269)

Search tool configuration. Defaults to `true`.

---

### addToolConfig?

```ts
optional addToolConfig?: boolean | MemoryAddToolConfig;
```

Defined in: [src/memory/types.ts:274](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/types.ts#L274)

Add tool configuration. Defaults to `false` (opt-in). `true` lets the tool write to all writable stores; pass a [MemoryAddToolConfig](/docs/api/typescript/MemoryAddToolConfig/index.md) with `stores` to restrict it to specific ones.

---

### injection?

```ts
optional injection?: boolean | MemoryInjectionConfig;
```

Defined in: [src/memory/types.ts:290](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/memory/types.ts#L290)

Memory context injection. Defaults to `true`. `true` uses the default injection settings; pass a [MemoryInjectionConfig](/docs/api/typescript/MemoryInjectionConfig/index.md) to customize retrieval, timing, and formatting; `false` disables it.

`true` is equivalent to:

```ts
{
  trigger: 'userTurn',          // inject only on a fresh user ask
  maxEntries: 5,                // retrieve and inject up to 5 entries
  // query:  the latest user text on a user turn, else the most recent assistant text
  // format: a <memory> block with one <entry source="STORE_NAME"> per result (content escaped)
}
```