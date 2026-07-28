Defined in: [src/types/messages.ts:70](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L70)

A message in a conversation between user and assistant. Each message has a role (user or assistant) and an array of content blocks.

## Implements

-   `JSONSerializable`<[`MessageData`](/docs/api/typescript/MessageData/index.md)\>

## Constructors

### Constructor

```ts
new Message(data): Message;
```

Defined in: [src/types/messages.ts:101](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L101)

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | { `role`: [`Role`](/docs/api/typescript/Role/index.md); `content`: [`ContentBlock`](/docs/api/typescript/ContentBlock/index.md)\[\]; `trackingId?`: `string`; `metadata?`: `MessageMetadata`; } |
| `data.role` | [`Role`](/docs/api/typescript/Role/index.md) |
| `data.content` | [`ContentBlock`](/docs/api/typescript/ContentBlock/index.md)\[\] |
| `data.trackingId?` | `string` |
| `data.metadata?` | `MessageMetadata` |

#### Returns

`Message`

## Properties

### type

```ts
readonly type: "message";
```

Defined in: [src/types/messages.ts:74](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L74)

Discriminator for message type.

---

### role

```ts
readonly role: Role;
```

Defined in: [src/types/messages.ts:79](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L79)

The role of the message sender.

---

### content

```ts
readonly content: ContentBlock[];
```

Defined in: [src/types/messages.ts:84](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L84)

Array of content blocks that make up this message.

---

### trackingId

```ts
readonly trackingId: string;
```

Defined in: [src/types/messages.ts:94](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L94)

Durable, stable UUID for the message, assigned at construction. Every Message has one — a caller-supplied id is preserved, otherwise a fresh UUID is minted (so callers do not normally set it; a caller supplying its own should use a UUID v4, e.g. `crypto.randomUUID()`). Survives session save/restore, and is stripped before model calls. Preserved when a message is copied or restored, so ids are unique within a conversation, but the same message carries the same id across sessions (copying another agent’s messages does not re-key them).

---

### metadata?

```ts
optional metadata?: MessageMetadata;
```

Defined in: [src/types/messages.ts:99](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L99)

Optional metadata, not sent to model providers.

## Methods

### fromMessageData()

```ts
static fromMessageData(data): Message;
```

Defined in: [src/types/messages.ts:115](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L115)

Creates a Message instance from MessageData.

#### Parameters

| Parameter | Type |
| --- | --- |
| `data` | [`MessageData`](/docs/api/typescript/MessageData/index.md) |

#### Returns

`Message`

---

### toJSON()

```ts
toJSON(): MessageData;
```

Defined in: [src/types/messages.ts:131](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L131)

Serializes the Message to a JSON-compatible MessageData object. Called automatically by JSON.stringify().

#### Returns

[`MessageData`](/docs/api/typescript/MessageData/index.md)

#### Implementation of

```ts
JSONSerializable.toJSON
```

---

### fromJSON()

```ts
static fromJSON(data): Message;
```

Defined in: [src/types/messages.ts:147](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L147)

Creates a Message instance from MessageData. Alias for fromMessageData for API consistency.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | [`MessageData`](/docs/api/typescript/MessageData/index.md) | MessageData to deserialize |

#### Returns

`Message`

Message instance

---

### clone()

```ts
clone(): Message;
```

Defined in: [src/types/messages.ts:154](https://github.com/strands-agents/harness-sdk/blob/fe4cbb9486566154b1f94e3ea3c6a85a2bd81f43/strands-ts/src/types/messages.ts#L154)

Creates a deep copy of this Message (round-trips through serialization).

#### Returns

`Message`