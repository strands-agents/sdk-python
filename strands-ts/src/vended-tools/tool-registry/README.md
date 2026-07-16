# Tool Registry Tool

A vended tool that lets an agent list, register, re-register, and unregister tools on its own registry at runtime.

Registration is limited to tools hosted on an already-connected `McpClient` the developer explicitly pre-approved when constructing the tool. The model picks servers by alias, never by raw URL; loading tools from disk or inline source is not supported. `update` and `delete` only affect tools this same instance registered, tracked in a per-agent `WeakMap` so two agents sharing one tool instance never see each other's dynamic tools. Tool names must match `/^[a-zA-Z_][a-zA-Z0-9_]{0,63}$/`, and a configurable `maxDynamicTools` cap (default thirty-two) bounds registry growth over a long conversation.

## Usage

```typescript
import { Agent, BedrockModel, McpClient } from '@strands-agents/sdk'
import { makeToolRegistry } from '@strands-agents/sdk/vended-tools/tool-registry'

const weather = new McpClient({ url: 'https://weather.example.com/mcp' })
await weather.connect()

const registryTool = makeToolRegistry({
  mcpClients: { weather },
})

const agent = new Agent({
  model: new BedrockModel({ region: 'us-east-1' }),
  tools: [registryTool],
})

// The model can now discover and pull in tools at runtime:
//   {"operation": "list"}
//   {"operation": "create", "toolName": "get_forecast",
//    "source": "weather", "remoteName": "getForecast"}
//   {"operation": "delete", "toolName": "get_forecast"}
```

## API

### `makeToolRegistry(options?)`

Options:

- `mcpClients` (`Record<string, McpClient>`, optional): allow-list of MCP servers the tool may pull tools from. Keys are stable aliases the model references in `source`. Values must already be connected. When omitted, the tool degrades to read-only.
- `maxDynamicTools` (`number`, default thirty-two): upper bound on the number of tools this instance may register on any single agent.
- `name` (`string`, default `"tool_registry"`), `description` (`string`): passed through to the tool spec.

### Operations

All four operations share a flat input schema; the `operation` field is a discriminator.

- `list`: returns `{ tools: RegisteredTool[]; dynamicCount: number; dynamicLimit: number }`. Each entry includes `registeredByToolRegistry: boolean`.
- `create`: requires `toolName`, `source`; `remoteName` defaults to `toolName`; `descriptionOverride` optional. Registers the resolved MCP tool under `toolName` on the agent's registry.
- `update`: same inputs as `create`. Rejects any `toolName` this tool_registry instance did not register.
- `delete`: requires `toolName`. Rejects any `toolName` this tool_registry instance did not register.
