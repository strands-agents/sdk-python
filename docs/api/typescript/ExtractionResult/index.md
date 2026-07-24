Defined in: [src/memory/extraction/types.ts:41](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/types.ts#L41)

A discrete entry produced by an [Extractor](/docs/api/typescript/Extractor/index.md), ready to be written to a store via its `add`.

## Properties

### content

```ts
content: string;
```

Defined in: [src/memory/extraction/types.ts:43](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/types.ts#L43)

The textual content of the entry.

---

### metadata?

```ts
optional metadata?: Record<string, JSONValue>;
```

Defined in: [src/memory/extraction/types.ts:45](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/types.ts#L45)

Optional metadata to associate with the entry.