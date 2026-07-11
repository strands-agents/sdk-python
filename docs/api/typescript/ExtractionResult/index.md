Defined in: [src/memory/extraction/types.ts:41](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/extraction/types.ts#L41)

A discrete entry produced by an [Extractor](/docs/api/typescript/Extractor/index.md), ready to be written to a store via its `add`.

## Properties

### content

```ts
content: string;
```

Defined in: [src/memory/extraction/types.ts:43](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/extraction/types.ts#L43)

The textual content of the entry.

---

### metadata?

```ts
optional metadata?: Record<string, JSONValue>;
```

Defined in: [src/memory/extraction/types.ts:45](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/memory/extraction/types.ts#L45)

Optional metadata to associate with the entry.