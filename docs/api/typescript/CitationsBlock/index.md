Defined in: [src/types/citations.ts:173](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L173)

Citations content block within a message. Returned by models when document citations are enabled. This is an output-only block — users do not construct these directly.

## Implements

-   [`CitationsBlockData`](/docs/api/typescript/CitationsBlockData/index.md)
-   `JSONSerializable`<{ `citations`: `Serialized`<[`CitationsBlockData`](/docs/api/typescript/CitationsBlockData/index.md)\>; }>

## Constructors

### Constructor

```ts
new CitationsBlock(data): CitationsBlock;
```

Defined in: [src/types/citations.ts:191](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L191)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`CitationsBlockData`](/docs/api/typescript/CitationsBlockData/index.md) |

#### Returns

`CitationsBlock`

## Properties

### type

```ts
readonly type: "citationsBlock";
```

Defined in: [src/types/citations.ts:179](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L179)

Discriminator for citations content.

---

### citations

```ts
readonly citations: Citation[];
```

Defined in: [src/types/citations.ts:184](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L184)

Array of citations linking generated content to source locations.

#### Implementation of

[`CitationsBlockData`](/docs/api/typescript/CitationsBlockData/index.md).[`citations`](/docs/api/typescript/CitationsBlockData/index.md#citations)

---

### content

```ts
readonly content: CitationGeneratedContent[];
```

Defined in: [src/types/citations.ts:189](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L189)

The generated content associated with these citations.

#### Implementation of

[`CitationsBlockData`](/docs/api/typescript/CitationsBlockData/index.md).[`content`](/docs/api/typescript/CitationsBlockData/index.md#content)

## Methods

### toJSON()

```ts
toJSON(): {
  citations: {
     citations: {
        location:   | {
           type: "documentChar";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "documentPage";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "documentChunk";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "searchResult";
           searchResultIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "web";
           url: string;
           domain?: string;
         };
        source: string;
        sourceContent: {
           text: string;
        }[];
        title: string;
     }[];
     content: {
        text: string;
     }[];
  };
};
```

Defined in: [src/types/citations.ts:200](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L200)

Serializes the CitationsBlock to a JSON-compatible ContentBlockData object. Called automatically by JSON.stringify().

#### Returns

```ts
{
  citations: {
     citations: {
        location:   | {
           type: "documentChar";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "documentPage";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "documentChunk";
           documentIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "searchResult";
           searchResultIndex: number;
           start: number;
           end: number;
         }
           | {
           type: "web";
           url: string;
           domain?: string;
         };
        source: string;
        sourceContent: {
           text: string;
        }[];
        title: string;
     }[];
     content: {
        text: string;
     }[];
  };
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `citations` | { `citations`: { `location`: | { `type`: `"documentChar"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentPage"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentChunk"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"searchResult"`; `searchResultIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"web"`; `url`: `string`; `domain?`: `string`; }; `source`: `string`; `sourceContent`: { `text`: `string`; }\[\]; `title`: `string`; }\[\]; `content`: { `text`: `string`; }\[\]; } | \- | [src/types/citations.ts:200](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L200) |
| `citations.citations` | { `location`: | { `type`: `"documentChar"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentPage"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentChunk"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"searchResult"`; `searchResultIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"web"`; `url`: `string`; `domain?`: `string`; }; `source`: `string`; `sourceContent`: { `text`: `string`; }\[\]; `title`: `string`; }\[\] | Array of citations linking generated content to source locations. | [src/types/citations.ts:160](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L160) |
| `citations.content` | { `text`: `string`; }\[\] | The generated content associated with these citations. | [src/types/citations.ts:165](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L165) |

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): CitationsBlock;
```

Defined in: [src/types/citations.ts:215](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/citations.ts#L215)

Creates a CitationsBlock instance from its wrapped data format.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | { `citations`: { `citations`: { `location`: | { `type`: `"documentChar"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentPage"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentChunk"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"searchResult"`; `searchResultIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"web"`; `url`: `string`; `domain?`: `string`; }; `source`: `string`; `sourceContent`: { `text`: `string`; }\[\]; `title`: `string`; }\[\]; `content`: { `text`: `string`; }\[\]; }; } | Wrapped CitationsBlockData to deserialize |
| `data.citations` | { `citations`: { `location`: | { `type`: `"documentChar"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentPage"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentChunk"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"searchResult"`; `searchResultIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"web"`; `url`: `string`; `domain?`: `string`; }; `source`: `string`; `sourceContent`: { `text`: `string`; }\[\]; `title`: `string`; }\[\]; `content`: { `text`: `string`; }\[\]; } | \- |
| `data.citations.citations` | { `location`: | { `type`: `"documentChar"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentPage"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"documentChunk"`; `documentIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"searchResult"`; `searchResultIndex`: `number`; `start`: `number`; `end`: `number`; } | { `type`: `"web"`; `url`: `string`; `domain?`: `string`; }; `source`: `string`; `sourceContent`: { `text`: `string`; }\[\]; `title`: `string`; }\[\] | Array of citations linking generated content to source locations. |
| `data.citations.content` | { `text`: `string`; }\[\] | The generated content associated with these citations. |

#### Returns

`CitationsBlock`

CitationsBlock instance