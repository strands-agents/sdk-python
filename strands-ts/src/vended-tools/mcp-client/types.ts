import type { JSONSchema, JSONValue } from '../../types/json.js'

/**
 * A tool exposed by a connected MCP server, as reported by `list_tools`.
 */
export interface McpClientToolSpec {
  /** Server-assigned tool name. */
  name: string
  /** Server-provided description shown to the model. Empty string when the server omits it. */
  description: string
  /** JSON Schema for the tool's input. */
  input_schema: JSONSchema
  /** Optional output schema, when the server publishes one. */
  output_schema?: JSONSchema
}

/**
 * Result of a successful `connect` call.
 */
export interface McpClientConnectResult {
  /** Opaque, unguessable session identifier. */
  session_id: string
  /** Canonicalised URL of the connected server. */
  server_url: string
}

/**
 * Result of a `list_tools` call.
 */
export interface McpClientListToolsResult {
  tools: McpClientToolSpec[]
}

/**
 * Result of a `call_tool` call.
 */
export interface McpClientCallToolResult {
  /** `'success'` when the server returned a non-error result, `'error'` otherwise. */
  status: 'success' | 'error'
  /** Concatenated text content, possibly truncated. */
  text: string
  /** Present and true when the returned text was truncated to the size cap. */
  truncated?: boolean
  /** Structured content when the server returned any. */
  structured_content?: JSONValue
  /** Present and true when the server flagged the tool call as an application-level error. */
  is_error?: boolean
}

/**
 * Result of a `disconnect` call.
 */
export interface McpClientDisconnectResult {
  disconnected: true
}

export type McpClientResult =
  McpClientConnectResult | McpClientListToolsResult | McpClientCallToolResult | McpClientDisconnectResult
