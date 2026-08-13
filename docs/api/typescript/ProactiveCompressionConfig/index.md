```ts
type ProactiveCompressionConfig = {
  compressionThreshold: number;
};
```

Defined in: [src/conversation-manager/conversation-manager.ts:55](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L55)

Configuration for proactive compression when passed as an object.

## Properties

### compressionThreshold

```ts
compressionThreshold: number;
```

Defined in: [src/conversation-manager/conversation-manager.ts:61](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/conversation-manager/conversation-manager.ts#L61)

Ratio of context window usage that triggers proactive compression. Value between 0 (exclusive) and 1 (inclusive). Defaults to 0.7 (compress when 70% of the context window is used).