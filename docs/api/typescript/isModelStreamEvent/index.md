```ts
function isModelStreamEvent(event): event is ModelStreamEvent;
```

Defined in: [src/models/streaming.ts:44](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/models/streaming.ts#L44)

Type guard to check if an event with a type discriminator is a ModelStreamEvent.

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `event` | { `type`: `string`; } | The event to check |
| `event.type` | `string` | \- |

## Returns

`event is ModelStreamEvent`

true if the event is a ModelStreamEvent