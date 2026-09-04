Defined in: [src/mcp/client.ts:88](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L88)

Filters controlling which MCP tools a client exposes.

## Properties

### allowed?

```ts
optional allowed?: McpToolMatcher[];
```

Defined in: [src/mcp/client.ts:90](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L90)

When present, only tools matching at least one matcher are exposed.

---

### rejected?

```ts
optional rejected?: McpToolMatcher[];
```

Defined in: [src/mcp/client.ts:92](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L92)

Tools matching at least one matcher are excluded, even when also allowed.