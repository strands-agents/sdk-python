```ts
type ElicitationCallback = (context, params) => Promise<ElicitResult>;
```

Defined in: [src/types/elicitation.ts:21](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/types/elicitation.ts#L21)

Callback invoked when an MCP server sends an elicitation request to gather user input during tool execution.

## Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `context` | [`ElicitationContext`](/docs/api/typescript/ElicitationContext/index.md) | Request context including abort signal. |
| `params` | `ElicitRequestParams` | The elicitation parameters from the server (message, requested schema or URL). |

## Returns

`Promise`<`ElicitResult`\>

The user’s response: accept (with content), decline, or cancel.