```ts
type McpTransport = Omit<Transport, "sessionId"> & {
  sessionId?: string;
};
```

Defined in: [src/mcp/client.ts:29](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L29)

Widened transport type that accepts MCP transport implementations without requiring explicit casts.

Under `exactOptionalPropertyTypes`, `StreamableHTTPClientTransport` is not directly assignable to `Transport` because its `sessionId` getter returns `string | undefined`, while `Transport` declares `sessionId?: string` (absent or string, but not explicitly undefined). This type relaxes that constraint so users can pass any MCP transport without `as Transport`.

## Type Declaration

| Name | Type | Defined in |
| --- | --- | --- |
| `sessionId?` | `string` | [src/mcp/client.ts:29](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L29) |