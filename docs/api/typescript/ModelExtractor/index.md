Defined in: [src/memory/extraction/model-extractor.ts:36](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/memory/extraction/model-extractor.ts#L36)

An [Extractor](/docs/api/typescript/Extractor/index.md) that calls a language model to distill messages into discrete facts.

Use for self-managed stores that hold plain text and want automatic distillation. Backends that extract server-side should omit the extractor entirely (the no-extractor passthrough hands them raw messages instead).

## Example

```typescript
extraction: {
  trigger: [new InvocationTrigger()],
  extractor: new ModelExtractor({ model: cheapModel, systemPrompt: 'Extract user preferences.' }),
}
```

## Implements

-   [`Extractor`](/docs/api/typescript/Extractor/index.md)

## Constructors

### Constructor

```ts
new ModelExtractor(options?): ModelExtractor;
```

Defined in: [src/memory/extraction/model-extractor.ts:40](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/memory/extraction/model-extractor.ts#L40)

#### Parameters

| Parameter | Type |
| --- | --- |
| `options` | [`ModelExtractorOptions`](/docs/api/typescript/ModelExtractorOptions/index.md) |

#### Returns

`ModelExtractor`

## Methods

### extract()

```ts
extract(messages, context?): Promise<ExtractionResult[]>;
```

Defined in: [src/memory/extraction/model-extractor.ts:47](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/memory/extraction/model-extractor.ts#L47)

Extract entries from a batch of messages.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | [`MessageData`](/docs/api/typescript/MessageData/index.md)\[\] | The filtered messages to extract from |
| `context?` | [`ExtractorContext`](/docs/api/typescript/ExtractorContext/index.md) | Optional context (e.g. a fallback model) |

#### Returns

`Promise`<[`ExtractionResult`](/docs/api/typescript/ExtractionResult/index.md)\[\]>

The entries to write to the store

#### Implementation of

[`Extractor`](/docs/api/typescript/Extractor/index.md).[`extract`](/docs/api/typescript/Extractor/index.md#extract)