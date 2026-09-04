Defined in: [src/mcp/client.ts:96](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L96)

Per-call overrides for [McpClient.listTools](/docs/api/typescript/McpClient/index.md#listtools).

## Properties

### prefix?

```ts
optional prefix?: string;
```

Defined in: [src/mcp/client.ts:98](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L98)

Prefix for agent-facing tool names. An empty string disables a prefix set on the client.

---

### toolFilters?

```ts
optional toolFilters?: McpToolFilters;
```

Defined in: [src/mcp/client.ts:100](https://github.com/strands-agents/harness-sdk/blob/e5b6b6dd62a46604b7c30b1f508f575018a3c443/strands-ts/src/mcp/client.ts#L100)

Tool filters. An empty object disables filters set on the client.