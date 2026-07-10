```ts
type CitationLocation =
  | {
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
```

Defined in: [src/types/citations.ts:14](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L14)

Discriminated union of citation location types. Each variant uses a `type` field to identify the location kind.

## Union Members

### Type Literal

```ts
{
  type: "documentChar";
  documentIndex: number;
  start: number;
  end: number;
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `type` | `"documentChar"` | Location referencing character positions within a document. | [src/types/citations.ts:19](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L19) |
| `documentIndex` | `number` | Index of the source document. | [src/types/citations.ts:24](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L24) |
| `start` | `number` | Start character position. | [src/types/citations.ts:29](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L29) |
| `end` | `number` | End character position. | [src/types/citations.ts:34](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L34) |

---

### Type Literal

```ts
{
  type: "documentPage";
  documentIndex: number;
  start: number;
  end: number;
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `type` | `"documentPage"` | Location referencing page positions within a document. | [src/types/citations.ts:40](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L40) |
| `documentIndex` | `number` | Index of the source document. | [src/types/citations.ts:45](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L45) |
| `start` | `number` | Start page number. | [src/types/citations.ts:50](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L50) |
| `end` | `number` | End page number. | [src/types/citations.ts:55](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L55) |

---

### Type Literal

```ts
{
  type: "documentChunk";
  documentIndex: number;
  start: number;
  end: number;
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `type` | `"documentChunk"` | Location referencing chunk positions within a document. | [src/types/citations.ts:61](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L61) |
| `documentIndex` | `number` | Index of the source document. | [src/types/citations.ts:66](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L66) |
| `start` | `number` | Start chunk index. | [src/types/citations.ts:71](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L71) |
| `end` | `number` | End chunk index. | [src/types/citations.ts:76](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L76) |

---

### Type Literal

```ts
{
  type: "searchResult";
  searchResultIndex: number;
  start: number;
  end: number;
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `type` | `"searchResult"` | Location referencing a search result. | [src/types/citations.ts:82](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L82) |
| `searchResultIndex` | `number` | Index of the search result. | [src/types/citations.ts:87](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L87) |
| `start` | `number` | Start position within the search result. | [src/types/citations.ts:92](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L92) |
| `end` | `number` | End position within the search result. | [src/types/citations.ts:97](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L97) |

---

### Type Literal

```ts
{
  type: "web";
  url: string;
  domain?: string;
}
```

| Name | Type | Description | Defined in |
| --- | --- | --- | --- |
| `type` | `"web"` | Location referencing a web URL. | [src/types/citations.ts:103](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L103) |
| `url` | `string` | The URL of the web source. | [src/types/citations.ts:108](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L108) |
| `domain?` | `string` | The domain of the web source. | [src/types/citations.ts:113](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/types/citations.ts#L113) |