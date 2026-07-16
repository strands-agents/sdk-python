/**
 * Shared types and constants for the tool_registry tool.
 */

import type { JSONSchema } from '../../types/json.js'

/**
 * Provider-accepted tool name: leading letter/underscore, then letters/digits/
 * underscore, capped at 64 characters. Stricter than the underlying registry
 * (which also allows `-`) because the dynamically-registered tool name is
 * echoed into user-visible spec strings; disallowing `-` keeps every generated
 * identifier a legal JavaScript identifier.
 */
export const TOOL_NAME_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]{0,63}$/

/** Maximum number of tools a single tool_registry instance may register. */
export const MAX_DYNAMIC_TOOLS = 32

export const TOOL_REGISTRY_DESCRIPTION =
  "Manage the agent's tool registry at runtime. Supports operations to `list` " +
  'currently registered tools, `create` a new binding to an already-connected ' +
  'MCP server tool, `update` an existing binding, and `delete` a previously ' +
  'registered binding. Registration is limited to remote MCP tools; loading ' +
  'tools from a file path or inline source is intentionally not supported.'

/** A tool entry in a `list` response. */
export interface RegisteredTool {
  /** The tool name as visible to the agent. */
  name: string
  /** The tool description as visible to the model. */
  description: string
  /** The tool's JSON input schema, if any. */
  inputSchema?: JSONSchema
  /**
   * True when this tool was registered via the tool_registry tool
   * (i.e. it can be updated/deleted by this tool). False for developer-registered tools.
   */
  registeredByToolRegistry: boolean
}

/** Result payload for the `list` operation. */
export interface ListResult {
  tools: RegisteredTool[]
  dynamicCount: number
  dynamicLimit: number
}

/** Result payload for `create`, `update`, and `delete` operations. */
export interface MutationResult {
  operation: 'create' | 'update' | 'delete'
  name: string
  dynamicCount: number
}

/**
 * Error thrown for validation failures inside the tool_registry tool.
 * Distinct class so callers can pattern-match on it separately from generic
 * ToolValidationError from the SDK registry.
 */
export class ToolRegistryError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ToolRegistryError'
  }
}
