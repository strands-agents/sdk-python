/**
 * Agent-callable MCP client tool.
 *
 * Lets an agent, at runtime, connect to a Model Context Protocol server on a
 * developer-set allowlist, list its tools, invoke one, and disconnect. Thin shim over
 * the SDK's `McpClient` — see `README.md` for the security model.
 */

export { makeMcpClient } from './mcp-client.js'
export type { MakeMcpClientOptions } from './mcp-client.js'
export type {
  McpClientCallToolResult,
  McpClientConnectResult,
  McpClientDisconnectResult,
  McpClientListToolsResult,
  McpClientResult,
  McpClientToolSpec,
} from './types.js'
