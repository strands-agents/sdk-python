Defined in: [src/types/media.ts:291](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L291)

Video content block.

## Implements

-   [`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md)
-   `JSONSerializable`<{ `video`: `Serialized`<[`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new VideoBlock(data): VideoBlock;
```

Defined in: [src/types/media.ts:307](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L307)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md) |

#### Returns

`VideoBlock`

## Properties

### type

```ts
readonly type: "videoBlock";
```

Defined in: [src/types/media.ts:295](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L295)

Discriminator for video content.

---

### format

```ts
readonly format: VideoFormat;
```

Defined in: [src/types/media.ts:300](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L300)

Video format.

#### Implementation of

[`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md).[`format`](/docs/api/typescript/VideoBlockData/index.md#format)

---

### source

```ts
readonly source: VideoSource;
```

Defined in: [src/types/media.ts:305](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L305)

Video source.

#### Implementation of

[`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md).[`source`](/docs/api/typescript/VideoBlockData/index.md#source)

## Methods

### toJSON()

```ts
toJSON(): {
  video: {
     format: VideoFormat;
     source:   | {
        bytes: string;
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      };
  };
};
```

Defined in: [src/types/media.ts:330](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L330)

Serializes the VideoBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify(). Uint8Array bytes are encoded as base64 string.

#### Returns

```ts
{
  video: {
     format: VideoFormat;
     source:   | {
        bytes: string;
      }
        | {
        location: {
           type: "s3";
           uri: string;
           bucketOwner?: string;
        };
      };
  };
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `video` | { `format`: [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md); `source`: | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; } | \- | [src/types/media.ts:330](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L330) |
| `video.format` | [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md) | Video format. | [src/types/media.ts:280](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L280) |
| `video.source` | | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Video source. | [src/types/media.ts:285](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L285) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): VideoBlock;
```

Defined in: [src/types/media.ts:352](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/media.ts#L352)

Creates a VideoBlock instance from its wrapped data format. Base64-encoded bytes are decoded back to Uint8Array.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `video`: { `format`: [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; }; } | Wrapped VideoBlockData to deserialize (accepts both string and Uint8Array for bytes) |
| `data.video` | { `format`: [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; } | \- |
| `data.video.format` | [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md) | Video format. |
| `data.video.source` | | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Video source. |

#### Returns

`VideoBlock`

VideoBlock instance