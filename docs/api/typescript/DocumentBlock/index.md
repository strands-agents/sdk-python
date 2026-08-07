Defined in: [src/types/media.ts:427](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L427)

Document content block.

## Implements

-   [`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md)
-   `JSONSerializable`<{ `document`: `Serialized`<[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new DocumentBlock(data): DocumentBlock;
```

Defined in: [src/types/media.ts:458](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L458)

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

Defined in: [src/types/media.ts:431](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L431)

Discriminator for document content.

---

### name

```ts
readonly name: string;
```

Defined in: [src/types/media.ts:436](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L436)

Document name.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`name`](/docs/api/typescript/DocumentBlockData/index.md#name)

---

### format

```ts
readonly format: DocumentFormat;
```

Defined in: [src/types/media.ts:441](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L441)

Document format.

#### Implementation of

[`DocumentBlockData`](/docs/api/typescript/DocumentBlockData/index.md).[`format`](/docs/api/typescript/DocumentBlockData/index.md#format)

---

### source

```ts
readonly source: DocumentSource;
```

Defined in: [src/types/media.ts:446](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L446)

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

Defined in: [src/types/media.ts:451](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L451)

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

Defined in: [src/types/media.ts:456](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L456)

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

Defined in: [src/types/media.ts:503](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L503)

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
| `document` | { `name`: `string`; `format`: [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md); `source`: | { `bytes`: `string`; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; `citations?`: { `enabled`: `boolean`; }; `context?`: `string`; } | \- | [src/types/media.ts:503](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L503) |
| `document.name` | `string` | Document name. | [src/types/media.ts:401](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L401) |
| `document.format` | [`DocumentFormat`](/docs/api/typescript/DocumentFormat/index.md) | Document format. | [src/types/media.ts:406](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L406) |
| `document.source` | | { `bytes`: `string`; } | { `text`: `string`; } | { `content`: { `text`: `string`; }\[\]; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Document source. | [src/types/media.ts:411](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L411) |
| `document.citations?` | { `enabled`: `boolean`; } | Citation configuration. | [src/types/media.ts:416](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L416) |
| `document.citations.enabled` | `boolean` | \- | [src/types/media.ts:416](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L416) |
| `document.context?` | `string` | Context information for the document. | [src/types/media.ts:421](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L421) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): DocumentBlock;
```

Defined in: [src/types/media.ts:532](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/media.ts#L532)

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