```ts
type InvokeArgs =
  | string
  | ContentBlock[]
  | ContentBlockData[]
  | Message[]
  | MessageData[]
  | InterruptResponseContent[]
  | InterruptResponseContentData[]
  | CheckpointResumeContent;
```

Defined in: [src/types/agent.ts:58](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/types/agent.ts#L58)

**`Experimental`**

Arguments for invoking an agent.

Supports multiple input formats:

-   `string` - User text input (wrapped in TextBlock, creates user Message)
-   `ContentBlock[]` | `ContentBlockData[]` - Array of content blocks (creates single user Message)
-   `Message[]` | `MessageData[]` - Array of messages (appends all to conversation)
-   `InterruptResponseContent[]` - Array of interrupt responses (resumes from interrupted state)
-   `CheckpointResumeContent` - Resume payload for a checkpointing agent

The `CheckpointResumeContent` member is experimental and subject to change.