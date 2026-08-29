Defined in: [src/types/messages.ts:476](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/types/messages.ts#L476)

Data for a reasoning block.

## Properties

### text?

```ts
optional text?: string;
```

Defined in: [src/types/messages.ts:480](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/types/messages.ts#L480)

The text content of the reasoning process.

---

### signature?

```ts
optional signature?: string;
```

Defined in: [src/types/messages.ts:485](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/types/messages.ts#L485)

A cryptographic signature for verification purposes.

---

### redactedContent?

```ts
optional redactedContent?: Uint8Array;
```

Defined in: [src/types/messages.ts:490](https://github.com/strands-agents/harness-sdk/blob/9062527eeda294e2f1381f385b36d0ba2fab9492/strands-ts/src/types/messages.ts#L490)

The redacted content of the reasoning process.