Defined in: [src/mcp/client.ts:47](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L47)

Configuration for MCP task-augmented tool execution.

WARNING: MCP Tasks is an experimental feature in both the MCP specification and this SDK. The API may change without notice in future versions.

When provided to McpClient, enables task-based tool invocation which supports long-running tools with progress tracking. Without this config, tools are called directly without task management.

## Properties

### ttl?

```ts
optional ttl?: number;
```

Defined in: [src/mcp/client.ts:52](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L52)

Time-to-live in milliseconds for task polling. Defaults to 60000 (60 seconds).

---

### pollTimeout?

```ts
optional pollTimeout?: number;
```

Defined in: [src/mcp/client.ts:58](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/client.ts#L58)

Maximum time in milliseconds to wait for task completion during polling. Defaults to 300000 (5 minutes).