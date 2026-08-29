Defined in: [src/memory/extraction/types.ts:74](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/memory/extraction/types.ts#L74)

Transforms conversation messages into discrete, searchable entries.

Implementations distill raw turns into facts worth remembering. Optional on a store’s [ExtractionConfig](/docs/api/typescript/ExtractionConfig/index.md): when absent, the manager passes messages straight to the store (see the no-extractor passthrough on [MemoryStore](/docs/api/typescript/MemoryStore/index.md)), which is the right path for backends that extract server-side.

## Methods

### extract()

```ts
extract(messages, context?): Promise<ExtractionResult[]>;
```

Defined in: [src/memory/extraction/types.ts:82](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/memory/extraction/types.ts#L82)

Extract entries from a batch of messages.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`MessageData`](/docs/api/typescript/MessageData/index.md)\[\] | The filtered messages to extract from |
| `context?` | [`ExtractorContext`](/docs/api/typescript/ExtractorContext/index.md) | Optional context (e.g. a fallback model) |

#### Returns

`Promise`<[`ExtractionResult`](/docs/api/typescript/ExtractionResult/index.md)\[\]>

The entries to write to the store