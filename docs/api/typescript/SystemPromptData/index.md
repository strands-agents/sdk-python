```ts
type SystemPromptData = string | SystemContentBlockData[];
```

Defined in: [src/types/messages.ts:751](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/types/messages.ts#L751)

Data representation of a system prompt. Can be a simple string or an array of system content block data for advanced caching.

This is the data interface counterpart to SystemPrompt, following the “Data” pattern.