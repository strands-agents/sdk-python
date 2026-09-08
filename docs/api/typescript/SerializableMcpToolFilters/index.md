Defined in: [src/mcp/config.ts:9](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/mcp/config.ts#L9)

Tool filters in a serializable form, for an MCP server config entry. Because a config file cannot carry a `RegExp`, each pattern is a string compiled to a regex, matched from the start of the server-side tool name.

## Properties

### allowed?

```ts
optional allowed?: string[];
```

Defined in: [src/mcp/config.ts:11](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/mcp/config.ts#L11)

When present, only tools whose names match one of these patterns are exposed.

---

### rejected?

```ts
optional rejected?: string[];
```

Defined in: [src/mcp/config.ts:13](https://github.com/strands-agents/harness-sdk/blob/ec6eb54a58382c3ebaced3e68489f3a0955c8319/strands-ts/src/mcp/config.ts#L13)

Tools whose names match one of these patterns are excluded, even when also allowed.