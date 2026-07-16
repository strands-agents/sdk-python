/**
 * Runtime tool-registry-management tool (CRUD over the agent's own tools).
 *
 * Only tools hosted on pre-approved MCP clients may be registered dynamically.
 * Loading tools from a file path or inline source is intentionally not
 * supported. See ../README.md for the design rationale.
 */

import { z } from 'zod'
import { tool } from '../../tools/tool-factory.js'
import { McpTool } from '../../tools/mcp-tool.js'
import { ToolValidationError } from '../../errors.js'
import type { LocalAgent } from '../../types/agent.js'
import type { McpClient } from '../../mcp/index.js'
import type { JSONSchema, JSONValue } from '../../types/json.js'
import {
  MAX_DYNAMIC_TOOLS,
  TOOL_NAME_PATTERN,
  TOOL_REGISTRY_DESCRIPTION,
  ToolRegistryError,
  type ListResult,
  type MutationResult,
  type RegisteredTool,
} from './types.js'

/**
 * Remembers which tool names this tool_registry instance registered on each
 * agent's `toolRegistry`. Keyed on the agent instance; each factory closes
 * over its own map, so:
 *
 *  - Two agents using the same tool_registry instance don't share ownership.
 *  - Two tool_registry factories in the same agent don't collide.
 *  - The set doesn't leak into the SDK registry's own Tool metadata (which
 *    other consumers of the registry might introspect).
 */
type OwnershipMap = WeakMap<LocalAgent, Set<string>>

function getOwnedNames(map: OwnershipMap, agent: LocalAgent): Set<string> {
  let owned = map.get(agent)
  if (!owned) {
    owned = new Set()
    map.set(agent, owned)
  }
  return owned
}

function validateToolName(name: string): void {
  if (typeof name !== 'string' || !TOOL_NAME_PATTERN.test(name)) {
    throw new ToolRegistryError(`invalid tool name '${name}': must match ${TOOL_NAME_PATTERN.source}`)
  }
}

/**
 * Locate a specific tool on an already-connected MCP client and adapt it to
 * a local name / description.
 */
async function resolveMcpTool(
  client: McpClient,
  remoteName: string,
  localName: string,
  descriptionOverride?: string
): Promise<McpTool> {
  const tools = await client.listTools()
  const found = tools.find((t) => t.name === remoteName)
  if (!found) {
    throw new ToolRegistryError(`tool '${remoteName}' not found on MCP server`)
  }
  return new McpTool({
    name: localName,
    description: descriptionOverride ?? found.description,
    // `McpTool`'s constructor requires an inputSchema; fall back to the empty
    // object schema when the remote tool doesn't declare one, rather than
    // casting away an `undefined`.
    inputSchema: (found.toolSpec.inputSchema ?? { type: 'object', properties: {} }) as JSONSchema,
    ...(found.toolSpec.outputSchema !== undefined && {
      outputSchema: found.toolSpec.outputSchema as JSONSchema,
    }),
    client,
  })
}

/**
 * Zod schema for input validation. The dispatch is enforced by `.refine`
 * (rather than a discriminated union) so the schema stays flat and the model
 * has a single, predictable set of parameters to fill in.
 *
 * This schema is the single source of truth for the tool's input shape. If a
 * field is added, renamed, or removed here, update the JSDoc on
 * `makeToolRegistry` and the README to match.
 */
const toolRegistryInputSchema = z
  .object({
    operation: z.enum(['list', 'create', 'update', 'delete']).describe('One of "list", "create", "update", "delete".'),
    toolName: z
      .string()
      .optional()
      .describe(
        "Local name for the tool on the agent's registry. Required for create, update, delete. " +
          'Must match ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$.'
      ),
    source: z
      .string()
      .optional()
      .describe('Alias of an MCP client pre-approved at tool construction time. Required for create/update.'),
    remoteName: z
      .string()
      .optional()
      .describe('Name of the tool on the MCP server pointed at by `source`. Defaults to `toolName`.'),
    descriptionOverride: z
      .string()
      .optional()
      .describe("Optional description to expose to the model in place of the MCP server's advertised description."),
  })
  .refine((v) => v.operation === 'list' || v.toolName !== undefined, {
    message: '`toolName` is required for create, update, and delete',
  })
  .refine((v) => (v.operation !== 'create' && v.operation !== 'update') || v.source !== undefined, {
    message: '`source` is required for create and update',
  })

/**
 * Options for {@link makeToolRegistry}.
 */
export interface MakeToolRegistryOptions {
  /**
   * Mapping from a stable, developer-chosen alias to an already-connected MCP
   * client. Only tools hosted on these clients may be registered dynamically.
   * When omitted or empty, `create` and `update` always error (the tool
   * degrades to a read-only view of the registry).
   */
  mcpClients?: Record<string, McpClient>
  /**
   * Upper bound on the number of tools this instance may add to the agent's
   * registry. Defaults to {@link MAX_DYNAMIC_TOOLS}.
   */
  maxDynamicTools?: number
  /** Tool name. Defaults to `"tool_registry"`. */
  name?: string
  /** Tool description shown to the model. */
  description?: string
}

/**
 * Create a runtime tool-registry-management tool bound to a set of MCP sources.
 *
 * The returned tool exposes four operations to the model:
 * - `list`: enumerate all tools currently on the agent's registry, flagging
 *   the ones this instance has registered itself.
 * - `create`: register a new tool that is a thin binding to a specific tool
 *   hosted on one of the pre-approved MCP clients.
 * - `update`: re-bind a previously registered tool.
 * - `delete`: unregister a tool that this same tool_registry instance
 *   registered. Developer-registered tools are never removable.
 *
 * @example
 * ```typescript
 * const weather = new McpClient({ url: 'http://weather.example/mcp' })
 * await weather.connect()
 * const registryTool = makeToolRegistry({ mcpClients: { weather } })
 * const agent = new Agent({ tools: [registryTool] })
 * ```
 */
export function makeToolRegistry(options: MakeToolRegistryOptions = {}): ReturnType<typeof tool> {
  const clients: Record<string, McpClient> = { ...(options.mcpClients ?? {}) }
  const maxDynamicTools = options.maxDynamicTools ?? MAX_DYNAMIC_TOOLS
  const toolName = options.name ?? 'tool_registry'
  const description = options.description ?? TOOL_REGISTRY_DESCRIPTION

  if (maxDynamicTools < 1) {
    throw new Error('maxDynamicTools must be at least 1')
  }

  const ownership: OwnershipMap = new WeakMap()

  return tool({
    name: toolName,
    description,
    inputSchema: toolRegistryInputSchema,
    callback: async (input, context): Promise<JSONValue> => {
      if (!context) {
        throw new Error('Tool context is required for tool_registry operations')
      }

      const agent = context.agent
      const registry = agent.toolRegistry
      const owned = getOwnedNames(ownership, agent)

      if (input.operation === 'list') {
        const tools: RegisteredTool[] = registry.list().map((t) => {
          const entry: RegisteredTool = {
            name: t.name,
            description: t.description,
            registeredByToolRegistry: owned.has(t.name),
          }
          if (t.toolSpec.inputSchema !== undefined) {
            entry.inputSchema = t.toolSpec.inputSchema
          }
          return entry
        })
        const result: ListResult = {
          tools,
          dynamicCount: owned.size,
          dynamicLimit: maxDynamicTools,
        }
        return result as unknown as JSONValue
      }

      // The schema refine guarantees `toolName` is present for create/update/
      // delete, but it does not reject the empty string. Coalesce to '' for
      // TS narrowing and let validateToolName reject empty and other
      // ill-formed names uniformly.
      const requestedName = input.toolName ?? ''
      validateToolName(requestedName)

      // Never allow this tool to remove or replace itself.
      if (requestedName === toolName) {
        throw new ToolRegistryError(`cannot ${input.operation} the tool_registry tool ('${requestedName}') itself`)
      }

      if (input.operation === 'delete') {
        if (!owned.has(requestedName)) {
          throw new ToolRegistryError(
            `tool '${requestedName}' was not registered via tool_registry; ` +
              'developer-registered tools cannot be removed'
          )
        }
        registry.remove(requestedName)
        owned.delete(requestedName)
        const result: MutationResult = {
          operation: 'delete',
          name: requestedName,
          dynamicCount: owned.size,
        }
        return result as unknown as JSONValue
      }

      // create / update below share the source-resolution logic.
      const source = input.source ?? ''
      if (!(source in clients)) {
        const allowed = Object.keys(clients).sort().join(', ') || '<none>'
        throw new ToolRegistryError(`unknown source '${source}': allowed sources are: ${allowed}`)
      }
      const client = clients[source]!
      const effectiveRemoteName = input.remoteName ?? requestedName

      if (input.operation === 'create') {
        if (registry.get(requestedName) !== undefined || owned.has(requestedName)) {
          throw new ToolRegistryError(`a tool named '${requestedName}' is already registered`)
        }
        if (owned.size >= maxDynamicTools) {
          throw new ToolRegistryError(
            `dynamic tool cap reached (${maxDynamicTools}); ` + 'delete an existing dynamically-registered tool first'
          )
        }
        // Reserve the slot synchronously so concurrent `create` calls in one
        // turn don't all pass the cap check before any of them writes.
        owned.add(requestedName)
        try {
          const newTool = await resolveMcpTool(client, effectiveRemoteName, requestedName, input.descriptionOverride)
          // A concurrent `delete` could have observed our reservation, treated
          // it as if the tool were already registered, and cleared it while we
          // were awaiting. If so, abort rather than resurrect an orphan entry
          // in the SDK registry that this instance no longer tracks (and thus
          // could never delete or update again).
          if (!owned.has(requestedName)) {
            throw new ToolRegistryError(
              `create of '${requestedName}' was cancelled by a concurrent delete before the tool could be registered`
            )
          }
          registry.add(newTool)
        } catch (err) {
          owned.delete(requestedName)
          if (err instanceof ToolValidationError) {
            throw new ToolRegistryError(err.message)
          }
          throw err
        }
        const result: MutationResult = {
          operation: 'create',
          name: requestedName,
          dynamicCount: owned.size,
        }
        return result as unknown as JSONValue
      }

      // operation === 'update'
      if (!owned.has(requestedName)) {
        throw new ToolRegistryError(
          `tool '${requestedName}' was not registered via tool_registry; ` +
            'developer-registered tools cannot be updated'
        )
      }
      const replacement = await resolveMcpTool(client, effectiveRemoteName, requestedName, input.descriptionOverride)
      // A concurrent `delete` could have observed our ownership, cleared it,
      // and removed the tool from the registry while we were awaiting the
      // MCP lookup. If so, abort rather than resurrect a tool the model
      // believed deleted (mirrors the create-during-delete guard).
      if (!owned.has(requestedName)) {
        throw new ToolRegistryError(
          `update of '${requestedName}' was cancelled by a concurrent delete before the tool could be re-bound`
        )
      }
      try {
        registry.addOrReplace([replacement])
      } catch (err) {
        if (err instanceof ToolValidationError) {
          throw new ToolRegistryError(err.message)
        }
        throw err
      }
      const result: MutationResult = {
        operation: 'update',
        name: requestedName,
        dynamicCount: owned.size,
      }
      return result as unknown as JSONValue
    },
  })
}
