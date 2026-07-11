```ts
type ElicitationCallback = (context, params) => Promise<ElicitResult>;
```

Defined in: [src/types/elicitation.ts:21](https://github.com/strands-agents/harness-sdk/blob/941d52513ea948b97c540e680c5cc8d6c0aeb54d/strands-ts/src/types/elicitation.ts#L21)

Callback invoked when an MCP server sends an elicitation request to gather user input during tool execution.

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`ElicitationContext`](/docs/api/typescript/ElicitationContext/index.md) | Request context including abort signal. |
| `params` | `ElicitRequestParams` | The elicitation parameters from the server (message, requested schema or URL). |

## Returns

`Promise`<`ElicitResult`\>

The user’s response: accept (with content), decline, or cancel.