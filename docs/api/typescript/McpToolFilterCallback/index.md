```ts
type McpToolFilterCallback = (tool) => boolean;
```

Defined in: [src/mcp/client.ts:79](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L79)

Decides whether a tool matches a filter. Receives the tool under its agent-facing name.

## Parameters

| Parameter | Type |
| --- | --- |
| `tool` | `McpTool` |

## Returns

`boolean`