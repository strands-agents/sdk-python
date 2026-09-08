Defined in: [src/types/messages.ts:217](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L217)

Text content block within a message.

## Implements

-   [`TextBlockData`](/docs/api/typescript/TextBlockData/index.md)
-   `JSONSerializable`<[`TextBlockData`](/docs/api/typescript/TextBlockData/index.md)\>

## Constructors

### Constructor

```ts
new TextBlock(data): TextBlock;
```

Defined in: [src/types/messages.ts:228](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L228)

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

Defined in: [src/types/messages.ts:221](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L221)

Discriminator for text content.

---

### text

```ts
readonly text: string;
```

Defined in: [src/types/messages.ts:226](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L226)

Plain text content.

#### Implementation of

[`TextBlockData`](/docs/api/typescript/TextBlockData/index.md).[`text`](/docs/api/typescript/TextBlockData/index.md#text)

## Methods

### toJSON()

```ts
toJSON(): TextBlockData;
```

Defined in: [src/types/messages.ts:236](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L236)

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

Defined in: [src/types/messages.ts:246](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/types/messages.ts#L246)

Creates a TextBlock instance from TextBlockData.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | [`TextBlockData`](/docs/api/typescript/TextBlockData/index.md) | TextBlockData to deserialize |

#### Returns

`TextBlock`

TextBlock instance