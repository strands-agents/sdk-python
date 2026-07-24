Defined in: [src/mcp/config.ts:10](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L10)

Configuration for a single MCP server entry in a config file or object.

Provide either `command` (stdio transport) or `url` (streamable-http/SSE), not both. When `transport` is omitted, it is auto-detected from the fields present.

## Properties

### command?

```ts
optional command?: string;
```

Defined in: [src/mcp/config.ts:12](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L12)

Command to spawn (stdio transport, supports `${VAR}` or `${env:VAR}` interpolation).

---

### args?

```ts
optional args?: string[];
```

Defined in: [src/mcp/config.ts:14](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L14)

Arguments passed to the command (supports `${VAR}` or `${env:VAR}` interpolation).

---

### env?

```ts
optional env?: Record<string, string>;
```

Defined in: [src/mcp/config.ts:16](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L16)

Environment variables passed to the child process (supports `${VAR}` or `${env:VAR}` interpolation).

---

### cwd?

```ts
optional cwd?: string;
```

Defined in: [src/mcp/config.ts:18](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L18)

Working directory for the spawned process (supports `${VAR}` or `${env:VAR}` interpolation).

---

### url?

```ts
optional url?: string;
```

Defined in: [src/mcp/config.ts:20](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L20)

Server endpoint URL (streamable-http or SSE transport, supports `${VAR}` or `${env:VAR}` interpolation).

---

### headers?

```ts
optional headers?: Record<string, string>;
```

Defined in: [src/mcp/config.ts:22](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L22)

HTTP headers sent with every request (supports `${VAR}` or `${env:VAR}` interpolation).

---

### transport?

```ts
optional transport?: "stdio" | "sse" | "streamable-http";
```

Defined in: [src/mcp/config.ts:24](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L24)

Explicit transport type. When omitted, auto-detected: `command` → stdio, `url` → streamable-http.

---

### auth?

```ts
optional auth?: McpClientCredentials;
```

Defined in: [src/mcp/config.ts:26](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L26)

Client credentials for OAuth machine-to-machine auth (streamable-http only).

---

### disabled?

```ts
optional disabled?: boolean;
```

Defined in: [src/mcp/config.ts:28](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L28)

When true, this server is skipped during loadServers.

---

### continueOnError?

```ts
optional continueOnError?: boolean;
```

Defined in: [src/mcp/config.ts:30](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L30)

When true, config or connection failures skip this server instead of throwing.

---

### tasksConfig?

```ts
optional tasksConfig?: TasksConfig;
```

Defined in: [src/mcp/config.ts:32](https://github.com/strands-agents/harness-sdk/blob/ec1c0db842d3a9a35c08f7a0b2dc132370baa0fa/strands-ts/src/mcp/config.ts#L32)

Task-augmented tool execution configuration (experimental).