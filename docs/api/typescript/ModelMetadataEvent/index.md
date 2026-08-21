Defined in: [src/models/streaming.ts:257](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L257)

Event containing metadata about the stream. Includes usage statistics, performance metrics, and trace information.

## Implements

-   [`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md)

## Constructors

### Constructor

```ts
new ModelMetadataEvent(data): ModelMetadataEvent;
```

Defined in: [src/models/streaming.ts:278](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L278)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md) |

#### Returns

`ModelMetadataEvent`

## Properties

### type

```ts
readonly type: "modelMetadataEvent";
```

Defined in: [src/models/streaming.ts:261](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L261)

Discriminator for metadata events.

#### Implementation of

[`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md).[`type`](/docs/api/typescript/ModelMetadataEventData/index.md#type)

---

### usage?

```ts
readonly optional usage?: Usage;
```

Defined in: [src/models/streaming.ts:266](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L266)

Token usage information.

#### Implementation of

[`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md).[`usage`](/docs/api/typescript/ModelMetadataEventData/index.md#usage)

---

### metrics?

```ts
readonly optional metrics?: Metrics;
```

Defined in: [src/models/streaming.ts:271](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L271)

Performance metrics.

#### Implementation of

[`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md).[`metrics`](/docs/api/typescript/ModelMetadataEventData/index.md#metrics)

---

### trace?

```ts
readonly optional trace?: unknown;
```

Defined in: [src/models/streaming.ts:276](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L276)

Trace information for observability.

#### Implementation of

[`ModelMetadataEventData`](/docs/api/typescript/ModelMetadataEventData/index.md).[`trace`](/docs/api/typescript/ModelMetadataEventData/index.md#trace)