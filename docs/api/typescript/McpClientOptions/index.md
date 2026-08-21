Defined in: [src/mcp/client.ts:104](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L104)

Behavioral options shared by all MCP client configurations.

## Extends

-   `RuntimeConfig`

## Properties

### applicationName?

```ts
optional applicationName?: string;
```

Defined in: [src/mcp/client.ts:33](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L33)

#### Inherited from

```ts
RuntimeConfig.applicationName
```

---

### applicationVersion?

```ts
optional applicationVersion?: string;
```

Defined in: [src/mcp/client.ts:34](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L34)

#### Inherited from

```ts
RuntimeConfig.applicationVersion
```

---

### disableMcpInstrumentation?

```ts
optional disableMcpInstrumentation?: boolean;
```

Defined in: [src/mcp/client.ts:106](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L106)

Disable OpenTelemetry MCP instrumentation.

---

### prefix?

```ts
optional prefix?: string;
```

Defined in: [src/mcp/client.ts:109](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L109)

Prefix for agent-facing tool names, applied as `<prefix>_<toolName>`.

---

### toolFilters?

```ts
optional toolFilters?: McpToolFilters;
```

Defined in: [src/mcp/client.ts:112](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L112)

Filters controlling which tools this client exposes.

---

### tasksConfig?

```ts
optional tasksConfig?: TasksConfig;
```

Defined in: [src/mcp/client.ts:119](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L119)

Configuration for task-augmented tool execution (experimental). When provided (even as empty object), enables MCP task-based tool invocation. When undefined, tools are called directly without task management.

---

### elicitationCallback?

```ts
optional elicitationCallback?: ElicitationCallback;
```

Defined in: [src/mcp/client.ts:126](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L126)

Callback to handle server-initiated elicitation requests. When provided, the client advertises elicitation support (form + url modes) and routes incoming elicitation requests to this callback.

---

### continueOnError?

```ts
optional continueOnError?: boolean;
```

Defined in: [src/mcp/client.ts:129](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L129)

When true, connection failures are logged as warnings instead of throwing.

---

### logHandler?

```ts
optional logHandler?: (params) => void;
```

Defined in: [src/mcp/client.ts:132](https://github.com/strands-agents/harness-sdk/blob/1fd743a7fe7cba5547824c6123aab7e6f3a7f10f/strands-ts/src/mcp/client.ts#L132)

Called when the server emits a log message. Defaults to routing through the Strands logger.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | `LoggingMessageNotificationParams` |

#### Returns

`void`