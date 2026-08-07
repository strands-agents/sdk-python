Defined in: [src/mcp/client.ts:88](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L88)

Filters controlling which MCP tools a client exposes.

## Properties

### allowed?

```ts
optional allowed?: McpToolMatcher[];
```

Defined in: [src/mcp/client.ts:90](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L90)

When present, only tools matching at least one matcher are exposed.

---

### rejected?

```ts
optional rejected?: McpToolMatcher[];
```

Defined in: [src/mcp/client.ts:92](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L92)

Tools matching at least one matcher are excluded, even when also allowed.