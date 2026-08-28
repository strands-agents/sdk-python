Defined in: [src/types/media.ts:102](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L102)

S3 location for media and document sources.

## Implements

-   [`S3LocationData`](/docs/api/typescript/S3LocationData/index.md)
-   `JSONSerializable`<[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md)\>

## Constructors

### Constructor

```ts
new S3Location(data): S3Location;
```

Defined in: [src/types/media.ts:107](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L107)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | `Omit`<[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md), `"type"`\> & { `type?`: `"s3"`; } |

#### Returns

`S3Location`

## Properties

### type

```ts
readonly type: "s3";
```

Defined in: [src/types/media.ts:103](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L103)

Location type — always “s3”.

#### Implementation of

[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md).[`type`](/docs/api/typescript/S3LocationData/index.md#type)

---

### uri

```ts
readonly uri: string;
```

Defined in: [src/types/media.ts:104](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L104)

S3 URI in format: s3://bucket-name/key-name

#### Implementation of

[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md).[`uri`](/docs/api/typescript/S3LocationData/index.md#uri)

---

### bucketOwner?

```ts
readonly optional bucketOwner?: string;
```

Defined in: [src/types/media.ts:105](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L105)

AWS account ID of the S3 bucket owner (12-digit). Required if the bucket belongs to another AWS account.

#### Implementation of

[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md).[`bucketOwner`](/docs/api/typescript/S3LocationData/index.md#bucketowner)

## Methods

### toJSON()

```ts
toJSON(): S3LocationData;
```

Defined in: [src/types/media.ts:118](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L118)

Serializes the S3Location to a JSON-compatible S3LocationData object. Called automatically by JSON.stringify().

#### Returns

[`S3LocationData`](/docs/api/typescript/S3LocationData/index.md)

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): S3Location;
```

Defined in: [src/types/media.ts:132](https://github.com/strands-agents/harness-sdk/blob/1941a726750c1659f10e010a1ea62106d31a24e3/strands-ts/src/types/media.ts#L132)

Creates an S3Location instance from S3LocationData.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | [`S3LocationData`](/docs/api/typescript/S3LocationData/index.md) | S3LocationData to deserialize |

#### Returns

`S3Location`

S3Location instance