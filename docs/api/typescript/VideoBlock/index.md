Defined in: [src/types/media.ts:391](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L391)

Video content block.

## Implements

-   [`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md)
-   `JSONSerializable`<{ `video`: `Serialized`<[`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new VideoBlock(data): VideoBlock;
```

Defined in: [src/types/media.ts:407](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L407)

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

Defined in: [src/types/media.ts:395](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L395)

Discriminator for video content.

---

### format

```ts
readonly format: VideoFormat;
```

Defined in: [src/types/media.ts:400](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L400)

Video format.

#### Implementation of

[`VideoBlockData`](/docs/api/typescript/VideoBlockData/index.md).[`format`](/docs/api/typescript/VideoBlockData/index.md#format)

---

### source

```ts
readonly source: VideoSource;
```

Defined in: [src/types/media.ts:405](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L405)

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

Defined in: [src/types/media.ts:430](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L430)

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
| `video` | { `format`: [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md); `source`: | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; } | \- | [src/types/media.ts:430](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L430) |
| `video.format` | [`VideoFormat`](/docs/api/typescript/VideoFormat/index.md) | Video format. | [src/types/media.ts:380](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L380) |
| `video.source` | | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Video source. | [src/types/media.ts:385](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L385) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): VideoBlock;
```

Defined in: [src/types/media.ts:452](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/media.ts#L452)

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