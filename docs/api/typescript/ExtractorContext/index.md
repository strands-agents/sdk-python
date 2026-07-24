Defined in: [src/memory/extraction/types.ts:55](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/types.ts#L55)

Context passed to [Extractor.extract](/docs/api/typescript/Extractor/index.md#extract).

Lets the manager hand an extractor a fallback model without the extractor having to be wired to the agent directly. [ModelExtractor](/docs/api/typescript/ModelExtractor/index.md) uses its own configured model when set, else [defaultModel](#defaultmodel).

## Properties

### defaultModel?

```ts
optional defaultModel?: Model;
```

Defined in: [src/memory/extraction/types.ts:57](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/memory/extraction/types.ts#L57)

The agent’s model, supplied so an extractor can default to it.