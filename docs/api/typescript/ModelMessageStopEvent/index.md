Defined in: [src/models/streaming.ts:204](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/streaming.ts#L204)

Event emitted when the message completes.

## Implements

-   [`ModelMessageStopEventData`](/docs/api/typescript/ModelMessageStopEventData/index.md)

## Constructors

### Constructor

```ts
new ModelMessageStopEvent(data): ModelMessageStopEvent;
```

Defined in: [src/models/streaming.ts:220](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/streaming.ts#L220)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`ModelMessageStopEventData`](/docs/api/typescript/ModelMessageStopEventData/index.md) |

#### Returns

`ModelMessageStopEvent`

## Properties

### type

```ts
readonly type: "modelMessageStopEvent";
```

Defined in: [src/models/streaming.ts:208](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/streaming.ts#L208)

Discriminator for message stop events.

#### Implementation of

[`ModelMessageStopEventData`](/docs/api/typescript/ModelMessageStopEventData/index.md).[`type`](/docs/api/typescript/ModelMessageStopEventData/index.md#type)

---

### stopReason

```ts
readonly stopReason: StopReason;
```

Defined in: [src/models/streaming.ts:213](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/streaming.ts#L213)

Reason why generation stopped.

#### Implementation of

[`ModelMessageStopEventData`](/docs/api/typescript/ModelMessageStopEventData/index.md).[`stopReason`](/docs/api/typescript/ModelMessageStopEventData/index.md#stopreason)

---

### additionalModelResponseFields?

```ts
readonly optional additionalModelResponseFields?: JSONValue;
```

Defined in: [src/models/streaming.ts:218](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/models/streaming.ts#L218)

Additional provider-specific response fields.

#### Implementation of

[`ModelMessageStopEventData`](/docs/api/typescript/ModelMessageStopEventData/index.md).[`additionalModelResponseFields`](/docs/api/typescript/ModelMessageStopEventData/index.md#additionalmodelresponsefields)