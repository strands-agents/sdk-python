Defined in: [src/memory/extraction/model-extractor.ts:14](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/extraction/model-extractor.ts#L14)

Options for [ModelExtractor](/docs/api/typescript/ModelExtractor/index.md).

## Properties

### model?

```ts
optional model?: Model;
```

Defined in: [src/memory/extraction/model-extractor.ts:16](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/extraction/model-extractor.ts#L16)

Model used to extract facts. Defaults to the agent’s own model; set a cheaper one to cut cost.

---

### systemPrompt?

```ts
optional systemPrompt?: string;
```

Defined in: [src/memory/extraction/model-extractor.ts:18](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/memory/extraction/model-extractor.ts#L18)

System prompt steering what counts as a fact. Defaults to a general fact-extraction prompt.