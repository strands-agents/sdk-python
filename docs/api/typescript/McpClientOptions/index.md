Defined in: [src/mcp/client.ts:79](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L79)

Behavioral options shared by all MCP client configurations.

## Extends

-   `RuntimeConfig`

## Properties

### applicationName?

```ts
optional applicationName?: string;
```

Defined in: [src/mcp/client.ts:33](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L33)

#### Inherited from

```ts
RuntimeConfig.applicationName
```

---

### applicationVersion?

```ts
optional applicationVersion?: string;
```

Defined in: [src/mcp/client.ts:34](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L34)

#### Inherited from

```ts
RuntimeConfig.applicationVersion
```

---

### disableMcpInstrumentation?

```ts
optional disableMcpInstrumentation?: boolean;
```

Defined in: [src/mcp/client.ts:81](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L81)

Disable OpenTelemetry MCP instrumentation.

---

### tasksConfig?

```ts
optional tasksConfig?: TasksConfig;
```

Defined in: [src/mcp/client.ts:88](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L88)

Configuration for task-augmented tool execution (experimental). When provided (even as empty object), enables MCP task-based tool invocation. When undefined, tools are called directly without task management.

---

### elicitationCallback?

```ts
optional elicitationCallback?: ElicitationCallback;
```

Defined in: [src/mcp/client.ts:95](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L95)

Callback to handle server-initiated elicitation requests. When provided, the client advertises elicitation support (form + url modes) and routes incoming elicitation requests to this callback.

---

### continueOnError?

```ts
optional continueOnError?: boolean;
```

Defined in: [src/mcp/client.ts:98](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L98)

When true, connection failures are logged as warnings instead of throwing.

---

### logHandler?

```ts
optional logHandler?: (params) => void;
```

Defined in: [src/mcp/client.ts:101](https://github.com/strands-agents/harness-sdk/blob/f4a8f9f50803682e6078624153dcff14818bc120/strands-ts/src/mcp/client.ts#L101)

Called when the server emits a log message. Defaults to routing through the Strands logger.

#### Parameters

| Parameter | Type |
| --- | --- |
| `params` | `LoggingMessageNotificationParams` |

#### Returns

`void`