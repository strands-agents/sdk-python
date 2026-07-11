Defined in: [src/models/streaming.ts:414](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/models/streaming.ts#L414)

Text delta within a content block. Represents incremental text content from the model.

## Properties

### type

```ts
type: "textDelta";
```

Defined in: [src/models/streaming.ts:418](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/models/streaming.ts#L418)

Discriminator for text delta.

---

### text

```ts
text: string;
```

Defined in: [src/models/streaming.ts:423](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/models/streaming.ts#L423)

Incremental text content.