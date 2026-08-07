Defined in: [src/models/streaming.ts:446](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L446)

Reasoning content delta within a content block. Represents incremental reasoning or thinking content.

## Properties

### type

```ts
type: "reasoningContentDelta";
```

Defined in: [src/models/streaming.ts:450](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L450)

Discriminator for reasoning delta.

---

### text?

```ts
optional text?: string;
```

Defined in: [src/models/streaming.ts:455](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L455)

Incremental reasoning text.

---

### signature?

```ts
optional signature?: string;
```

Defined in: [src/models/streaming.ts:460](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L460)

Incremental signature data.

---

### redactedContent?

```ts
optional redactedContent?: Uint8Array;
```

Defined in: [src/models/streaming.ts:465](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/models/streaming.ts#L465)

Incremental redacted content data.