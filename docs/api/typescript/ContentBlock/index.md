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

Defined in: [src/types/messages.ts:190](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/types/messages.ts#L190)