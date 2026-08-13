```ts
type ContentBlock =
  | TextBlock
  | ToolUseBlock
  | ToolResultBlock
  | ReasoningBlock
  | CachePointBlock
  | GuardContentBlock
  | ImageBlock
  | VideoBlock
  | DocumentBlock
  | CitationsBlock;
```

Defined in: [src/types/messages.ts:190](https://github.com/strands-agents/harness-sdk/blob/333885bbdabbb126ca305af4cc30d2214417e39c/strands-ts/src/types/messages.ts#L190)