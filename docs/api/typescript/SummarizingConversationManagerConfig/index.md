```ts
type SummarizingConversationManagerConfig = {
  model?: Model;
  summaryRatio?: number;
  preserveRecentMessages?: number;
  summarizationSystemPrompt?: string;
  proactiveCompression?: boolean | ProactiveCompressionConfig;
  pinFirst?: number;
};
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:28](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L28)

Configuration for the summarization conversation manager.

## Properties

### model?

```ts
optional model?: Model;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:34](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L34)

Model to use for generating summaries. When provided, overrides the model attached to the agent. Useful when you want to use a different model than the one attached to the agent.

---

### summaryRatio?

```ts
optional summaryRatio?: number;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:40](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L40)

Ratio of messages to summarize when context overflow occurs. Value is clamped to \[0.1, 0.8\]. Defaults to 0.3 (summarize 30% of oldest messages).

---

### preserveRecentMessages?

```ts
optional preserveRecentMessages?: number;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:46](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L46)

Minimum number of recent messages to always keep. Defaults to 10.

---

### summarizationSystemPrompt?

```ts
optional summarizationSystemPrompt?: string;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:52](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L52)

Custom system prompt for summarization. If not provided, uses a default prompt that produces structured bullet-point summaries.

---

### proactiveCompression?

```ts
optional proactiveCompression?: boolean | ProactiveCompressionConfig;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:61](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L61)

Enable proactive context compression before the model call.

-   `true`: compress when 70% of the context window is used (default threshold).
-   `{ compressionThreshold: number }`: compress at the specified ratio (0, 1\].
-   `false` or omitted: disabled, only reactive overflow recovery is used.

---

### pinFirst?

```ts
optional pinFirst?: number;
```

Defined in: [src/conversation-manager/summarizing-conversation-manager.ts:67](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/conversation-manager/summarizing-conversation-manager.ts#L67)

Number of messages at the start of the conversation to permanently pin. Pinned messages are protected from summarization and compacted to the front.