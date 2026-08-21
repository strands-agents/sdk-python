Defined in: [src/models/streaming.ts:172](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L172)

Event emitted when a content block completes.

## Implements

-   `ModelContentBlockStopEventData`

## Constructors

### Constructor

```ts
new ModelContentBlockStopEvent(_data): ModelContentBlockStopEvent;
```

Defined in: [src/models/streaming.ts:178](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L178)

#### Parameters

| Parameter | Type |
| --- | --- |
| `_data` | `ModelContentBlockStopEventData` |

#### Returns

`ModelContentBlockStopEvent`

## Properties

### type

```ts
readonly type: "modelContentBlockStopEvent";
```

Defined in: [src/models/streaming.ts:176](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/models/streaming.ts#L176)

Discriminator for content block stop events.

#### Implementation of

```ts
ModelContentBlockStopEventData.type
```