Defined in: [src/models/streaming.ts:172](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L172)

Event emitted when a content block completes.

## Implements

-   `ModelContentBlockStopEventData`

## Constructors

### Constructor

```ts
new ModelContentBlockStopEvent(_data): ModelContentBlockStopEvent;
```

Defined in: [src/models/streaming.ts:178](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L178)

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

Defined in: [src/models/streaming.ts:176](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/models/streaming.ts#L176)

Discriminator for content block stop events.

#### Implementation of

```ts
ModelContentBlockStopEventData.type
```