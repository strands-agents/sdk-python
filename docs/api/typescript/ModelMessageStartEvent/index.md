Defined in: [src/models/streaming.ts:66](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/streaming.ts#L66)

Event emitted when a new message starts in the stream.

## Implements

-   [`ModelMessageStartEventData`](/docs/api/typescript/ModelMessageStartEventData/index.md)

## Constructors

### Constructor

```ts
new ModelMessageStartEvent(data): ModelMessageStartEvent;
```

Defined in: [src/models/streaming.ts:77](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/streaming.ts#L77)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`ModelMessageStartEventData`](/docs/api/typescript/ModelMessageStartEventData/index.md) |

#### Returns

`ModelMessageStartEvent`

## Properties

### type

```ts
readonly type: "modelMessageStartEvent";
```

Defined in: [src/models/streaming.ts:70](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/streaming.ts#L70)

Discriminator for message start events.

#### Implementation of

[`ModelMessageStartEventData`](/docs/api/typescript/ModelMessageStartEventData/index.md).[`type`](/docs/api/typescript/ModelMessageStartEventData/index.md#type)

---

### role

```ts
readonly role: Role;
```

Defined in: [src/models/streaming.ts:75](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/models/streaming.ts#L75)

The role of the message being started.

#### Implementation of

[`ModelMessageStartEventData`](/docs/api/typescript/ModelMessageStartEventData/index.md).[`role`](/docs/api/typescript/ModelMessageStartEventData/index.md#role)