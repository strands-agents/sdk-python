Defined in: [src/types/media.ts:527](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L527)

Document content block.

## Implements

-   [`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md)
-   `JSONSerializable`<{ `document`: `Serialized`<[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new DocumentBlock(data): DocumentBlock;
```

Defined in: [src/types/media.ts:558](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L558)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md) |

#### Returns

`DocumentBlock`

## Properties

### type

```ts
readonly type: "documentBlock";
```

Defined in: [src/types/media.ts:531](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L531)

Discriminator for document content.

---

### name

```ts
readonly name: string;
```

Defined in: [src/types/media.ts:536](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L536)

Document name.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`name`](/docs/api/typescript/DocumentBlockData/index.md#name)

---

### format

```ts
readonly format: DocumentFormat;
```

Defined in: [src/types/media.ts:541](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L541)

Document format.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`format`](/docs/api/typescript/DocumentBlockData/index.md#format)

---

### source

```ts
readonly source: DocumentSource;
```

Defined in: [src/types/media.ts:546](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L546)

Document source.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`source`](/docs/api/typescript/DocumentBlockData/index.md#source)

---

### citations?

```ts
readonly optional citations?: {
  enabled: boolean;
};
```

Defined in: [src/types/media.ts:551](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L551)

Citation configuration.

#### enabled

```ts
enabled: boolean;
```

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`citations`](/docs/api/typescript/DocumentBlockData/index.md#citations)

---

### context?

```ts
readonly optional context?: string;
```

Defined in: [src/types/media.ts:556](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L556)

Context information for the document.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`context`](/docs/api/typescript/DocumentBlockData/index.md#context)

## Methods

### toJSON()

```ts
toJSON(): {
  document: {
     name: string;
     format: DocumentFormat;
     source:   | {
        bytes: string;
      }
        | {
        text: string;
      }
        | {
        content: {
           text: string;
        }[];
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      };
     citations?: {
        enabled: boolean;
     };
     context?: string;
  };
};
```

Defined in: [src/types/media.ts:603](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L603)

Serializes the DocumentBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify(). Uint8Array bytes are encoded as base64 string.

#### Returns

```ts
{
  document: {
     name: string;
     format: DocumentFormat;
     source:   | {
        bytes: string;
      }
        | {
        text: string;
      }
        | {
        content: {
           text: string;
        }[];
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      };
     citations?: {
        enabled: boolean;
     };
     context?: string;
  };
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `document` | { `name`: `string`; `format`: [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md); `source`: | { `bytes`: `string`; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; `citations?`: { `enabled`: `boolean`; }; `context?`: `string`; } | \- | [src/types/media.ts:603](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L603) |
| `document.name` | `string` | Document name. | [src/types/media.ts:501](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L501) |
| `document.format` | [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md) | Document format. | [src/types/media.ts:506](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L506) |
| `document.source` | | { `bytes`: `string`; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Document source. | [src/types/media.ts:511](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L511) |
| `document.citations?` | { `enabled`: `boolean`; } | Citation configuration. | [src/types/media.ts:516](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L516) |
| `document.citations.enabled` | `boolean` | \- | [src/types/media.ts:516](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L516) |
| `document.context?` | `string` | Context information for the document. | [src/types/media.ts:521](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L521) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): DocumentBlock;
```

Defined in: [src/types/media.ts:632](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/types/media.ts#L632)

Creates a DocumentBlock instance from its wrapped data format. Base64-encoded bytes are decoded back to Uint8Array.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `document`: { `name`: `string`; `format`: [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; `citations?`: { `enabled`: `boolean`; }; `context?`: `string`; }; } | Wrapped DocumentBlockData to deserialize (accepts both string and Uint8Array for bytes) |
| `data.document` | { `name`: `string`; `format`: [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; `citations?`: { `enabled`: `boolean`; }; `context?`: `string`; } | \- |
| `data.document.name` | `string` | Document name. |
| `data.document.format` | [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md) | Document format. |
| `data.document.source` | | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Document source. |
| `data.document.citations?` | { `enabled`: `boolean`; } | Citation configuration. |
| `data.document.citations.enabled` | `boolean` | \- |
| `data.document.context?` | `string` | Context information for the document. |

#### Returns

`DocumentBlock`

DocumentBlock instance