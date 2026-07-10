Defined in: [src/models/streaming.ts:324](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/streaming.ts#L324)

Data for a redact event. Emitted when guardrails block content and redaction is enabled.

## Properties

### type

```ts
type: "modelRedactionEvent";
```

Defined in: [src/models/streaming.ts:328](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/streaming.ts#L328)

Discriminator for redact events.

---

### inputRedaction?

```ts
optional inputRedaction?: RedactInputContent;
```

Defined in: [src/models/streaming.ts:333](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/streaming.ts#L333)

Input redaction information (when input is blocked).

---

### outputRedaction?

```ts
optional outputRedaction?: RedactOutputContent;
```

Defined in: [src/models/streaming.ts:338](https://github.com/strands-agents/harness-sdk/blob/dad124e5b0c50916073da7d22f040371c09628ef/strands-ts/src/models/streaming.ts#L338)

Output redaction information (when output is blocked).