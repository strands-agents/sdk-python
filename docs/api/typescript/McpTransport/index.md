```ts
type McpTransport = Omit<Transport, "sessionId"> & {
  sessionId?: string;
};
```

Defined in: [src/mcp/client.ts:29](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L29)

Widened transport type that accepts MCP transport implementations without requiring explicit casts.

Under `exactOptionalPropertyTypes`, `StreamableHTTPClientTransport` is not directly assignable to `Transport` because its `sessionId` getter returns `string | undefined`, while `Transport` declares `sessionId?: string` (absent or string, but not explicitly undefined). This type relaxes that constraint so users can pass any MCP transport without `as Transport`.

## Type Declaration

| Name | Type | Defined in |
| --- | --- | --- |
| `sessionId?` | `string` | [src/mcp/client.ts:29](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L29) |