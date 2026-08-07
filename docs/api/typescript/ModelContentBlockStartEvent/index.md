Defined in: [src/models/streaming.ts:101](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L101)

Event emitted when a new content block starts in the stream.

## Implements

-   [`ModelContentBlockStartEventData`](/docs/api/typescript/ModelContentBlockStartEventData/index.md)

## Constructors

### Constructor

```ts
new ModelContentBlockStartEvent(data): ModelContentBlockStartEvent;
```

Defined in: [src/models/streaming.ts:113](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L113)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`ModelContentBlockStartEventData`](/docs/api/typescript/ModelContentBlockStartEventData/index.md) |

#### Returns

`ModelContentBlockStartEvent`

## Properties

### type

```ts
readonly type: "modelContentBlockStartEvent";
```

Defined in: [src/models/streaming.ts:105](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L105)

Discriminator for content block start events.

#### Implementation of

[`ModelContentBlockStartEventData`](/docs/api/typescript/ModelContentBlockStartEventData/index.md).[`type`](/docs/api/typescript/ModelContentBlockStartEventData/index.md#type)

---

### start?

```ts
readonly optional start?: ToolUseStart;
```

Defined in: [src/models/streaming.ts:111](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L111)

Information about the content block being started. Only present for tool use blocks.

#### Implementation of

[`ModelContentBlockStartEventData`](/docs/api/typescript/ModelContentBlockStartEventData/index.md).[`start`](/docs/api/typescript/ModelContentBlockStartEventData/index.md#start)