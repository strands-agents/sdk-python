Defined in: [src/types/messages.ts:215](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L215)

Text content block within a message.

## Implements

-   [`TextBlockData`](/docs/api/typescript/TextBlockData/index.md)
-   `JSONSerializable`<[`TextBlockData`](/docs/api/typescript/TextBlockData/index.md)\>

## Constructors

### Constructor

```ts
new TextBlock(data): TextBlock;
```

Defined in: [src/types/messages.ts:226](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L226)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | `string` |

#### Returns

`TextBlock`

## Properties

### type

```ts
readonly type: "textBlock";
```

Defined in: [src/types/messages.ts:219](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L219)

Discriminator for text content.

---

### text

```ts
readonly text: string;
```

Defined in: [src/types/messages.ts:224](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L224)

Plain text content.

#### Implementation of

[`TextBlockData`](/docs/api/typescript/TextBlockData/index.md).[`text`](/docs/api/typescript/TextBlockData/index.md#text)

## Methods

### toJSON()

```ts
toJSON(): TextBlockData;
```

Defined in: [src/types/messages.ts:234](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L234)

Serializes the TextBlock to a JSON-compatible TextBlockData object. Called automatically by JSON.stringify().

#### Returns

[`TextBlockData`](/docs/api/typescript/TextBlockData/index.md)

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): TextBlock;
```

Defined in: [src/types/messages.ts:244](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L244)

Creates a TextBlock instance from TextBlockData.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | [`TextBlockData`](/docs/api/typescript/TextBlockData/index.md) | TextBlockData to deserialize |

#### Returns

`TextBlock`

TextBlock instance