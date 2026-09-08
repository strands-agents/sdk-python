```ts
type SlidingWindowConversationManagerConfig = {
  windowSize?: number;
  shouldTruncateResults?: boolean;
  proactiveCompression?: boolean | ProactiveCompressionConfig;
  pinFirst?: number;
};
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:87](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L87)

Configuration for the sliding window conversation manager.

## Properties

### windowSize?

```ts
optional windowSize?: number;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:92](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L92)

Maximum number of messages to keep in the conversation history. Defaults to 40 messages.

---

### shouldTruncateResults?

```ts
optional shouldTruncateResults?: boolean;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:98](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L98)

Whether to truncate tool results when a message is too large for the model’s context window. Defaults to true.

---

### proactiveCompression?

```ts
optional proactiveCompression?: boolean | ProactiveCompressionConfig;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:107](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L107)

Enable proactive context compression before the model call.

-   `true`: compress when 70% of the context window is used (default threshold).
-   `{ compressionThreshold: number }`: compress at the specified ratio (0, 1\].
-   `false` or omitted: disabled, only reactive overflow recovery is used.

---

### pinFirst?

```ts
optional pinFirst?: number;
```

Defined in: [src/conversation-manager/sliding-window-conversation-manager.ts:113](https://github.com/strands-agents/harness-sdk/blob/57d070582e9bd93afb549b4d2f976ce0dbaf03b6/strands-ts/src/conversation-manager/sliding-window-conversation-manager.ts#L113)

Number of messages at the start of the conversation to permanently pin. Pinned messages are protected from eviction during context reduction.