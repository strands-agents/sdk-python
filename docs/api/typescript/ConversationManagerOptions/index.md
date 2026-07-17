```ts
type ConversationManagerOptions = {
  proactiveCompression?: boolean | ProactiveCompressionConfig;
};
```

Defined in: [src/conversation-manager/conversation-manager.ts:71](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/conversation-manager.ts#L71)

Configuration options for the ConversationManager base class.

## Properties

### proactiveCompression?

```ts
optional proactiveCompression?: boolean | ProactiveCompressionConfig;
```

Defined in: [src/conversation-manager/conversation-manager.ts:79](https://github.com/strands-agents/harness-sdk/blob/f0c6d6ece7af9c800e05f60eaf274134e45c7f47/strands-ts/src/conversation-manager/conversation-manager.ts#L79)

Enable proactive context compression before the model call.

-   `true`: compress when 70% of the context window is used (default threshold).
-   `{ compressionThreshold: number }`: compress at the specified ratio (0, 1\].
-   `false` or omitted: disabled, only reactive overflow recovery is used.