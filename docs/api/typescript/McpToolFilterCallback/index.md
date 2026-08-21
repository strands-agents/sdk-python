```ts
type McpToolFilterCallback = (tool) => boolean;
```

Defined in: [src/mcp/client.ts:79](https://github.com/strands-agents/harness-sdk/blob/f6765ab42296e23ac7af7c93e0ff3626fd09a60c/strands-ts/src/mcp/client.ts#L79)

Decides whether a tool matches a filter. Receives the tool under its agent-facing name.

## Parameters

| Parameter | Type |
| --- | --- |
| `tool` | `McpTool` |

## Returns

`boolean`