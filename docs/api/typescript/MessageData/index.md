Defined in: [src/types/messages.ts:35](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L35)

Data for a message.

## Properties

### role

```ts
role: Role;
```

Defined in: [src/types/messages.ts:39](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L39)

The role of the message sender.

---

### content

```ts
content: ContentBlockData[];
```

Defined in: [src/types/messages.ts:44](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L44)

Array of content blocks that make up this message.

---

### trackingId?

```ts
optional trackingId?: string;
```

Defined in: [src/types/messages.ts:50](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L50)

Durable, stable UUID for the message. Optional here because serialized/legacy data may omit it; the Message constructor backfills a fresh UUID when absent, so a live Message always has one.

---

### metadata?

```ts
optional metadata?: MessageMetadata;
```

Defined in: [src/types/messages.ts:55](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L55)

Optional metadata, not sent to model providers.