Defined in: [src/types/media.ts:272](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L272)

Image content block.

## Implements

-   [`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md)
-   `JSONSerializable`<{ `image`: `Serialized`<[`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new ImageBlock(data): ImageBlock;
```

Defined in: [src/types/media.ts:288](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L288)

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

Defined in: [src/types/media.ts:276](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L276)

Discriminator for image content.

---

### format

```ts
readonly format: "png" | "jpeg" | "jpg" | "gif" | "webp";
```

Defined in: [src/types/media.ts:281](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L281)

Image format.

#### Implementation of

[`ImageBlockData`](/docs/api/typescript/ImageBlockData/index.md).[`format`](/docs/api/typescript/ImageBlockData/index.md#format)

---

### source

```ts
readonly source: ImageSource;
```

Defined in: [src/types/media.ts:286](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L286)

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

Defined in: [src/types/media.ts:320](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L320)

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
| `image` | { `format`: `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"`; `source`: | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; }; } | \- | [src/types/media.ts:320](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L320) |
| `image.format` | `"png"` | `"jpeg"` | `"jpg"` | `"gif"` | `"webp"` | Image format. | [src/types/media.ts:261](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L261) |
| `image.source` | | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | { `url`: `string`; } | Image source. | [src/types/media.ts:266](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L266) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): ImageBlock;
```

Defined in: [src/types/media.ts:344](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/media.ts#L344)

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