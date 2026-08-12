Defined in: [src/types/messages.ts:655](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L655)

JSON content block within a message. Used for structured data returned from tools or model responses.

## Implements

-   `JsonBlockData`
-   `JSONSerializable`<`JsonBlockData`\>

## Constructors

### Constructor

```ts
new JsonBlock(data): JsonBlock;
```

Defined in: [src/types/messages.ts:666](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L666)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | `JsonBlockData` |

#### Returns

`JsonBlock`

## Properties

### type

```ts
readonly type: "jsonBlock";
```

Defined in: [src/types/messages.ts:659](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L659)

Discriminator for JSON content.

---

### json

```ts
readonly json: JSONValue;
```

Defined in: [src/types/messages.ts:664](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L664)

Structured JSON data.

#### Implementation of

```ts
JsonBlockData.json
```

## Methods

### toJSON()

```ts
toJSON(): JsonBlockData;
```

Defined in: [src/types/messages.ts:674](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L674)

Serializes the JsonBlock to a JSON-compatible JsonBlockData object. Called automatically by JSON.stringify().

#### Returns

`JsonBlockData`

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): JsonBlock;
```

Defined in: [src/types/messages.ts:684](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/types/messages.ts#L684)

Creates a JsonBlock instance from JsonBlockData.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | `JsonBlockData` | JsonBlockData to deserialize |

#### Returns

`JsonBlock`

JsonBlock instance