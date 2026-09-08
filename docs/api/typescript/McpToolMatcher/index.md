```ts
type McpToolMatcher = string | RegExp | McpToolFilterCallback;
```

Defined in: [src/mcp/client.ts:85](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/mcp/client.ts#L85)

Matches a tool for filtering. A string matches the server-side tool name exactly; a `RegExp` matches it from the start (as Python’s `Pattern.match` does); a callback receives the tool.