Defined in: [src/memory/types.ts:179](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/memory/types.ts#L179)

Options for MemoryManager.add.

## Properties

### metadata?

```ts
optional metadata?: Record<string, JSONValue>;
```

Defined in: [src/memory/types.ts:181](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/memory/types.ts#L181)

Metadata to associate with the added entry.

---

### stores?

```ts
optional stores?: string[];
```

Defined in: [src/memory/types.ts:183](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/memory/types.ts#L183)

Filter to specific writable stores by name. Omit to write to all writable stores.