```ts
type McpToolFilterCallback = (tool) => boolean;
```

Defined in: [src/mcp/client.ts:79](https://github.com/strands-agents/harness-sdk/blob/11ad6366a1578d432ea4cd2c3ed41b610953d297/strands-ts/src/mcp/client.ts#L79)

Decides whether a tool matches a filter. Receives the tool under its agent-facing name.

## Parameters

| Parameter | Type |
| --- | --- |
| `tool` | `McpTool` |

## Returns

`boolean`