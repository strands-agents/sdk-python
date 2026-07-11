Defined in: [src/mcp/client.ts:123](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L123)

MCP Client for interacting with Model Context Protocol servers.

## Constructors

### Constructor

```ts
new McpClient(args): McpClient;
```

Defined in: [src/mcp/client.ts:159](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L159)

#### Parameters

| Parameter | Type |
| --- | --- |
| `args` | [`McpClientConfig`](/docs/api/typescript/McpClientConfig/index.md) |

#### Returns

`McpClient`

## Properties

### DEFAULT\_TTL

```ts
readonly static DEFAULT_TTL: 60000 = 60000;
```

Defined in: [src/mcp/client.ts:125](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L125)

Default TTL for task polling in milliseconds (60 seconds).

---

### DEFAULT\_POLL\_TIMEOUT

```ts
readonly static DEFAULT_POLL_TIMEOUT: 300000 = 300000;
```

Defined in: [src/mcp/client.ts:128](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L128)

Default poll timeout for task completion in milliseconds (5 minutes).

## Accessors

### client

#### Get Signature

```ts
get client(): Client;
```

Defined in: [src/mcp/client.ts:228](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L228)

##### Returns

`Client`

---

### serverCapabilities

#### Get Signature

```ts
get serverCapabilities(): any;
```

Defined in: [src/mcp/client.ts:232](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L232)

##### Returns

`any`

---

### serverVersion

#### Get Signature

```ts
get serverVersion(): any;
```

Defined in: [src/mcp/client.ts:236](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L236)

##### Returns

`any`

---

### serverInstructions

#### Get Signature

```ts
get serverInstructions(): string;
```

Defined in: [src/mcp/client.ts:240](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L240)

##### Returns

`string`

---

### connectionState

#### Get Signature

```ts
get connectionState(): McpConnectionState;
```

Defined in: [src/mcp/client.ts:244](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L244)

##### Returns

[`McpConnectionState`](/docs/api/typescript/McpConnectionState/index.md)

---

### clientName

#### Get Signature

```ts
get clientName(): string;
```

Defined in: [src/mcp/client.ts:248](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L248)

##### Returns

`string`

---

### continueOnError

#### Get Signature

```ts
get continueOnError(): boolean;
```

Defined in: [src/mcp/client.ts:252](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L252)

##### Returns

`boolean`

---

### onToolsChanged

#### Set Signature

```ts
set onToolsChanged(callback): void;
```

Defined in: [src/mcp/client.ts:355](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L355)

Sets a callback invoked when the MCP server’s tool list changes at runtime.

##### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `callback` | (`oldTools`, `newTools`) => `void` | Handler receiving the previous tool names and the refreshed tool instances, or undefined to remove the callback. |

##### Returns

`void`

## Methods

### loadServers()

```ts
static loadServers(config, defaults?): Promise<McpClient[]>;
```

Defined in: [src/mcp/client.ts:137](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L137)

Parses an MCP servers config (file path or object) and returns McpClient instances.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `config` | | `string` | `Record`<`string`, [`McpServerConfig`](/docs/api/typescript/McpServerConfig/index.md)\> | A file path to a JSON config, or a flat server map object. |
| `defaults?` | [`McpClientOptions`](/docs/api/typescript/McpClientOptions/index.md) | Options applied to all clients unless overridden per-server. |

#### Returns

`Promise`<`McpClient`\[\]>

An array of McpClient instances ready to be passed to an Agent.

---

### connect()

```ts
connect(reconnect?): Promise<void>;
```

Defined in: [src/mcp/client.ts:266](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L266)

Connects the MCP client to the server.

Called lazily before any operation that requires a connection. When `continueOnError` is true, connection failures are swallowed and the client enters a `'failed'` state — subsequent calls are no-ops until `connect(true)` is called explicitly to retry.

#### Parameters

| Parameter | Type | Default value | Description |
| --- | --- | --- | --- |
| `reconnect` | `boolean` | `false` | When true, forces a reconnect even if already connected or failed. |

#### Returns

`Promise`<`void`\>

A promise that resolves when the connection is established.

---

### disconnect()

```ts
disconnect(): Promise<void>;
```

Defined in: [src/mcp/client.ts:298](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L298)

Disconnects the MCP client from the server and cleans up resources.

#### Returns

`Promise`<`void`\>

A promise that resolves when the disconnection is complete.

---

### \[asyncDispose\]()

```ts
asyncDispose: Promise<void>;
```

Defined in: [src/mcp/client.ts:309](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L309)

Enables the `await using` pattern for automatic resource cleanup. Delegates to [McpClient.disconnect](#disconnect).

#### Returns

`Promise`<`void`\>

---

### listTools()

```ts
listTools(): Promise<McpTool[]>;
```

Defined in: [src/mcp/client.ts:318](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L318)

Lists the tools available on the server and returns them as executable McpTool instances.

#### Returns

`Promise`<`McpTool`\[\]>

A promise that resolves with an array of McpTool instances.

---

### callTool()

```ts
callTool(
   tool,
   args,
options?): Promise<JSONValue>;
```

Defined in: [src/mcp/client.ts:393](https://github.com/strands-agents/harness-sdk/blob/e579bdedc73f07ea3b980a1268c57585bf984fe3/strands-ts/src/mcp/client.ts#L393)

Invoke a tool on the connected MCP server using an McpTool instance.

When `tasksConfig` was provided to the client constructor, uses experimental task-based invocation which supports long-running tools with progress tracking. Otherwise, calls tools directly without task management.

#### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `tool` | `McpTool` | The McpTool instance to invoke. |
| `args` | [`JSONValue`](/docs/api/typescript/JSONValue/index.md) | The arguments to pass to the tool. |
| `options?` | [`McpCallToolOptions`](/docs/api/typescript/McpCallToolOptions/index.md) | Optional settings for the request. |

#### Returns

`Promise`<[`JSONValue`](/docs/api/typescript/JSONValue/index.md)\>

A promise that resolves with the result of the tool invocation.