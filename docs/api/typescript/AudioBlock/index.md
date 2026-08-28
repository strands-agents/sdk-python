Defined in: [src/types/media.ts:162](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L162)

Audio content block.

## Implements

-   [`AudioBlockData`](/docs/api/typescript/AudioBlockData/index.md)
-   `JSONSerializable`<{ `audio`: `Serialized`<[`AudioBlockData`](/docs/api/typescript/AudioBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new AudioBlock(data): AudioBlock;
```

Defined in: [src/types/media.ts:172](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L172)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`AudioBlockData`](/docs/api/typescript/AudioBlockData/index.md) |

#### Returns

`AudioBlock`

## Properties

### type

```ts
readonly type: "audioBlock";
```

Defined in: [src/types/media.ts:164](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L164)

Discriminator for audio content.

---

### format

```ts
readonly format: AudioFormat;
```

Defined in: [src/types/media.ts:167](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L167)

Audio format.

#### Implementation of

[`AudioBlockData`](/docs/api/typescript/AudioBlockData/index.md).[`format`](/docs/api/typescript/AudioBlockData/index.md#format)

---

### source

```ts
readonly source: AudioSource;
```

Defined in: [src/types/media.ts:170](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L170)

Audio source.

#### Implementation of

[`AudioBlockData`](/docs/api/typescript/AudioBlockData/index.md).[`source`](/docs/api/typescript/AudioBlockData/index.md#source)

## Methods

### toJSON()

```ts
toJSON(): {
  audio: {
     format: AudioFormat;
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

Defined in: [src/types/media.ts:200](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L200)

Serializes the AudioBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify(). Uint8Array bytes are encoded as a base64 string.

#### Returns

```ts
{
  audio: {
     format: AudioFormat;
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

Wrapped audio block data

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `audio` | { `format`: [`AudioFormat`](/docs/api/typescript/AudioFormat/index.md); `source`: | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; } | \- | [src/types/media.ts:200](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L200) |
| `audio.format` | [`AudioFormat`](/docs/api/typescript/AudioFormat/index.md) | Audio format. | [src/types/media.ts:153](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L153) |
| `audio.source` | | { `bytes`: `string`; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Audio source. | [src/types/media.ts:156](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L156) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): AudioBlock;
```

Defined in: [src/types/media.ts:221](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L221)

Creates an AudioBlock instance from its wrapped data format. Base64-encoded bytes are decoded back to Uint8Array.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `audio`: { `format`: [`AudioFormat`](/docs/api/typescript/AudioFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; }; } | Wrapped AudioBlockData to deserialize |
| `data.audio` | { `format`: [`AudioFormat`](/docs/api/typescript/AudioFormat/index.md); `source`: | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; }; } | \- |
| `data.audio.format` | [`AudioFormat`](/docs/api/typescript/AudioFormat/index.md) | Audio format. |
| `data.audio.source` | | { `bytes`: `string` | `Uint8Array`<`ArrayBufferLike`\>; } | { `location`: { `type`: `"s3"`; `uri`: `string`; `bucketOwner?`: `string`; }; } | Audio source. |

#### Returns

`AudioBlock`

AudioBlock instance