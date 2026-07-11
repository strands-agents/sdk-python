Defined in: [src/types/messages.ts:474](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L474)

Data for a reasoning block.

## Properties

### text?

```ts
optional text?: string;
```

Defined in: [src/types/messages.ts:478](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L478)

The text content of the reasoning process.

---

### signature?

```ts
optional signature?: string;
```

Defined in: [src/types/messages.ts:483](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L483)

A cryptographic signature for verification purposes.

---

### redactedContent?

```ts
optional redactedContent?: Uint8Array;
```

Defined in: [src/types/messages.ts:488](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/messages.ts#L488)

The redacted content of the reasoning process.