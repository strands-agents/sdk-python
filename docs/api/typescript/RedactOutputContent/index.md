Defined in: [src/models/streaming.ts:307](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/streaming.ts#L307)

Information about output content redaction. May include the original content if captured during streaming.

## Properties

### redactedContent?

```ts
optional redactedContent?: string;
```

Defined in: [src/models/streaming.ts:312](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/streaming.ts#L312)

The original content that was blocked by guardrails. May not be available for all providers.

---

### replaceContent

```ts
replaceContent: string;
```

Defined in: [src/models/streaming.ts:317](https://github.com/strands-agents/harness-sdk/blob/a2ac1180f3709b565df82695f03d489f3bce5955/strands-ts/src/models/streaming.ts#L317)

The content to replace the redacted output with.