Defined in: [src/types/media.ts:172](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L172)

Image content block.

## Implements

-   [`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md)
-   `JSONSerializable`<{ `image`: `Serialized`<[`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new ImageBlock(data): ImageBlock;
```

Defined in: [src/types/media.ts:188](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L188)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md) |

#### Returns

`ImageBlock`

## Properties

### type

```ts
readonly type: "imageBlock";
```

Defined in: [src/types/media.ts:176](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L176)

Discriminator for image content.

---

### format

```ts
readonly format: "png" | "jpeg" | "jpg" | "gif" | "webp";
```

Defined in: [src/types/media.ts:181](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L181)

Image format.

#### Implementation of

[`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md).[`format`](/docs/api/typescript/ImageBlockData/index.md#format)

---

### source

```ts
readonly source: ImageSource;
```

Defined in: [src/types/media.ts:186](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L186)

Image source.

#### Implementation of

[`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md).[`source`](/docs/api/typescript/ImageBlockData/index.md#source)

## Methods

### toJSON()

```ts
toJSON(): {
  image: {
     format: "png" | "jpeg" | "jpg" | "gif" | "webp";
     source:   | {
        bytes: string;
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      }
        | {
        url: string;
      };
  };
};
```

Defined in: [src/types/media.ts:220](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L220)

Serializes the ImageBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify(). Uint8Array bytes are encoded as base64 string.

#### Returns

```ts
{
  image: {
     format: "png" | "jpeg" | "jpg" | "gif" | "webp";
     source:   | {
        bytes: string;
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      }
        | {
        url: string;
      };
  };
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `image` | { `format`: `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"`; `source`: | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; }; } | \- | [src/types/media.ts:220](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L220) |
| `image.format` | `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"` | Image format. | [src/types/media.ts:161](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L161) |
| `image.source` | | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; } | Image source. | [src/types/media.ts:166](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L166) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): ImageBlock;
```

Defined in: [src/types/media.ts:244](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/media.ts#L244)

Creates an ImageBlock instance from its wrapped data format. Base64-encoded bytes are decoded back to Uint8Array.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `image`: { `format`: `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"`; `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; }; }; } | Wrapped ImageBlockData to deserialize (accepts both string and Uint8Array for bytes) |
| `data.image` | { `format`: `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"`; `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; }; } | \- |
| `data.image.format` | `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"` | Image format. |
| `data.image.source` | | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; } | Image source. |

#### Returns

`ImageBlock`

ImageBlock instance