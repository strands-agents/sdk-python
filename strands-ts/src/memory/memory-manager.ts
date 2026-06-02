import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Tool } from '../tools/tool.js'
import type {
  MemoryEntry,
  MemoryManagerConfig,
  MemorySearchOptions,
  MemoryStore,
  MemoryAddOptions,
  MemoryToolConfig,
} from './types.js'
import type { JSONValue } from '../types/json.js'
import { tool } from '../tools/tool-factory.js'
import { z } from 'zod'
import { logger } from '../logging/logger.js'
import { normalizeError } from '../errors.js'

const SEARCH_TOOL_DESCRIPTION =
  'Search long-term memory for facts, preferences, or context from previous conversations. Use when you need background about the user or topic that may have been discussed before.'

const ADD_TOOL_DESCRIPTION =
  'Add facts, preferences, or decisions to long-term memory so they are remembered across conversations. Use when the user shares something worth recalling later.'

/**
 * Default maximum results per store when neither the caller nor the store specifies one.
 * Resolved by the {@link MemoryManager}.
 */
export const DEFAULT_MAX_SEARCH_RESULTS = 3

/** Flattens nested AggregateErrors so the leaves are concrete reasons, not errors-of-errors. */
function _flattenReasons(reasons: unknown[]): unknown[] {
  return reasons.flatMap((reason) => (reason instanceof AggregateError ? _flattenReasons(reason.errors) : [reason]))
}

/**
 * Provides cross-session memory retrieval and storage for agents.
 *
 * Manages one or more {@link MemoryStore} backends, exposing `search_memory` and
 * `add_memory` tools for agent-driven recall and persistence. Any tools the stores
 * themselves provide (via {@link MemoryStore.getTools}) are registered alongside these.
 *
 * @example
 * ```typescript
 * import { Agent, MemoryManager } from '@strands-agents/sdk'
 *
 * // Config shorthand
 * const agent = new Agent({
 *   model,
 *   memoryManager: { stores: [myStore], addToolConfig: true },
 * })
 *
 * // Class instance (for programmatic access)
 * const memoryManager = new MemoryManager({ stores: [myStore], addToolConfig: true })
 * const agent = new Agent({ model, memoryManager })
 * await memoryManager.search('user preferences')
 * ```
 */
export class MemoryManager implements Plugin {
  readonly name = 'strands:memory-manager'
  private readonly _config: MemoryManagerConfig
  private readonly _searchStores: MemoryStore[]
  private readonly _addStores: MemoryStore[]
  private readonly _searchToolConfig: MemoryToolConfig | false
  private readonly _addToolConfig: MemoryToolConfig | false
  private readonly _awaitWrites: boolean

  constructor(config: MemoryManagerConfig) {
    if (config.stores.length === 0) {
      throw new Error('MemoryManager: at least one store is required')
    }

    const seenNames = new Set<string>()
    for (const store of config.stores) {
      if (seenNames.has(store.name)) {
        throw new Error(`MemoryManager: duplicate store name '${store.name}'`)
      }
      seenNames.add(store.name)

      if (store.writable && !store.add) {
        throw new Error(`MemoryManager: store '${store.name}' is writable but has no add method`)
      }
    }

    this._config = config
    this._searchStores = config.stores
    this._addStores = config.stores.filter((s) => s.writable)
    this._awaitWrites = config.awaitWrites ?? false

    this._searchToolConfig =
      config.searchToolConfig === false
        ? false
        : typeof config.searchToolConfig === 'object'
          ? config.searchToolConfig
          : {}

    if (config.addToolConfig === undefined || config.addToolConfig === false) {
      this._addToolConfig = false
    } else {
      if (this._addStores.length === 0) {
        throw new Error('MemoryManager: addToolConfig is enabled but no stores are writable')
      }
      this._addToolConfig = typeof config.addToolConfig === 'object' ? config.addToolConfig : {}
    }
  }

  /**
   * Initializes the plugin with the agent.
   *
   * No lifecycle hooks are registered in this version.
   *
   * @param _agent - The agent this plugin is being attached to
   */
  initAgent(_agent: LocalAgent): void {}

  /**
   * Returns tools registered by this plugin.
   *
   * Includes the manager's own `search_memory` / `add_memory` tools (per their config) plus any
   * tools the configured stores expose via {@link MemoryStore.getTools}.
   *
   * @returns Array of tools to register with the agent
   */
  getTools(): Tool[] {
    const tools: Tool[] = []

    if (this._searchToolConfig !== false) {
      tools.push(this._createSearchTool(this._searchToolConfig))
    }

    if (this._addToolConfig !== false) {
      tools.push(this._createAddTool(this._addToolConfig))
    }

    for (const store of this._config.stores) {
      const storeTools = store.getTools?.() ?? []
      tools.push(...storeTools)
    }

    return tools
  }

  /**
   * Search stores for entries matching the query. If `stores` is provided, only searches to those named stores.
   *
   * This method is unscoped with full access to all configured stores.
   * Tool-level store scoping is applied by the search tool callback.
   * When `options.stores` is omitted, all stores are searched.
   *
   * Only `maxSearchResults` and routing (`stores`) cross this layer. Store-specific search
   * parameters (e.g. a Bedrock metadata `filter` or search-type override) are not expressible here
   * across heterogeneous stores — set them as per-instance defaults on the store, or call the
   * store's own `search()` directly for full control. Per-instance store policy (such as a tenant
   * filter) always applies, including when reached through the `search_memory` tool.
   *
   * @param query - The search query string
   * @param options - Optional max results (forwarded to all stores) and store name filter
   * @returns Array of memory entries from matching stores
   */
  async search(query: string, options?: MemorySearchOptions): Promise<MemoryEntry[]> {
    logger.debug(
      `query=<${query}>, max_search_results=<${options?.maxSearchResults}>, stores=<${options?.stores}> | searching stores`
    )

    const targetStores =
      options?.stores !== undefined
        ? [...new Set(options.stores)].map((name) => {
            const found = this._config.stores.find((s) => s.name === name)
            if (!found) {
              throw new Error(`MemoryManager: store '${name}' not found`)
            }
            return found
          })
        : this._config.stores

    const settled = await Promise.allSettled(
      targetStores.map((store) =>
        store.search(query, {
          maxSearchResults: options?.maxSearchResults ?? store.maxSearchResults ?? DEFAULT_MAX_SEARCH_RESULTS,
        })
      )
    )

    const results: MemoryEntry[] = []
    for (let i = 0; i < settled.length; i++) {
      const settledResult = settled[i]!
      const storeName = targetStores[i]!.name
      if (settledResult.status === 'rejected') {
        logger.warn(
          `store=<${storeName}>, reason=<${normalizeError(settledResult.reason).message}> | store search failed`
        )
        continue
      }
      for (const entry of settledResult.value) {
        // Stamp provenance so callers can tell which store produced each result.
        results.push({ ...entry, storeName })
      }
    }

    logger.debug(`results=<${results.length}> | search complete`)
    return results
  }

  /**
   * Add content to writable stores. If `stores` is provided, only writes to those named stores.
   *
   * This method is unscoped, with full access to all configured writable stores.
   * Tool-level store scoping is applied by the add tool callback.
   * When `options.stores` is omitted, all writable stores are targeted.
   *
   * Target stores are always validated synchronously (an unknown or read-only named store throws
   * immediately). The store writes themselves follow `awaitWrites` (resolved from
   * {@link MemoryAddOptions.awaitWrites} then {@link MemoryManagerConfig.awaitWrites}.
   * - fire-and-forget (default): resolves once writes are dispatched; per-store failures are logged.
   * - awaited: resolves after all writes settle and throws an `AggregateError` if any store fails.
   *
   * @param content - The text content to add
   * @param options - Optional metadata, store name filter, and per-call `awaitWrites` override
   */
  async add(content: string, options?: MemoryAddOptions): Promise<void> {
    let writableStores: MemoryStore[]

    if (options?.stores !== undefined) {
      writableStores = [...new Set(options.stores)].map((name) => {
        const found = this._config.stores.find((s) => s.name === name)
        if (!found) {
          throw new Error(`MemoryManager: store '${name}' not found`)
        }
        if (!found.writable) {
          throw new Error(`MemoryManager: store '${name}' is read-only`)
        }
        return found
      })
    } else {
      writableStores = this._addStores
    }

    if (writableStores.length === 0) {
      throw new Error('MemoryManager: no writable store matched')
    }

    const write = this._writeToStores(writableStores, content, options?.metadata)

    if (options?.awaitWrites ?? this._awaitWrites) {
      await write
    } else {
      // Fire-and-forget: failures are already logged inside _writeToStores; swallow the rejection
      // here so the detached promise never surfaces as an unhandled rejection.
      write.catch(() => {})
    }
  }

  /**
   * Writes content to every given store, logging per-store failures. Throws an `AggregateError` if
   * any store fails. Callers decide whether to await (observe failures) or fire-and-forget.
   */
  private async _writeToStores(
    stores: MemoryStore[],
    content: string,
    metadata: Record<string, JSONValue> | undefined
  ): Promise<void> {
    const settled = await Promise.allSettled(stores.map((store) => store.add!(content, metadata)))

    const failures: { store: string; reason: unknown }[] = []
    for (let i = 0; i < settled.length; i++) {
      const settledResult = settled[i]!
      if (settledResult.status === 'rejected') {
        const storeName = stores[i]!.name
        logger.warn(
          `store=<${storeName}>, reason=<${normalizeError(settledResult.reason).message}> | store write failed`
        )
        failures.push({ store: storeName, reason: settledResult.reason })
      }
    }
    if (failures.length > 0) {
      throw new AggregateError(
        failures.map((failure) => failure.reason),
        `MemoryManager: store writes failed: ${failures.map((failure) => failure.store).join(', ')}`
      )
    }
  }

  /**
   * Resolves the store names that a tool callback should target against the tool's scoped set.
   *
   * - Omitting `requested` targets all scoped stores.
   * - Names that are in scope are kept; out-of-scope names are dropped with a warning.
   * - When every requested name is out of scope, throws so the model receives an actionable error
   *   (the tool layer turns the thrown error into a model-visible result it can correct from).
   *
   * @param scopedNames - Store names available to this tool
   * @param requested - Store names the model asked for, if any
   * @returns A non-empty list of in-scope store names to target
   */
  private _resolveToolTargets(scopedNames: string[], requested?: string[]): string[] {
    if (requested === undefined || requested.length === 0) {
      return scopedNames
    }

    const inScope = requested.filter((name) => scopedNames.includes(name))
    const outOfScope = requested.filter((name) => !scopedNames.includes(name))

    if (inScope.length === 0) {
      throw new Error(
        `MemoryManager: requested=<${requested.join(', ')}> | none of the requested memory stores are available; available stores: ${scopedNames.join(', ')}`
      )
    }

    if (outOfScope.length > 0) {
      logger.warn(`requested=<${outOfScope.join(', ')}> | ignoring memory stores outside this tool's scope`)
    }

    return inScope
  }

  private _createSearchTool(config: MemoryToolConfig): Tool {
    let description = config.description ?? SEARCH_TOOL_DESCRIPTION
    const storeDescriptions = this._searchStores
      .filter((s) => s.description)
      .map((s) => `- ${s.name}: ${s.description}`)
    if (storeDescriptions.length > 0) {
      description += `\n\nAvailable memory stores:\n${storeDescriptions.join('\n')}`
      description +=
        '\n\nYou can target one or more memory stores by name if you know which domains are relevant, or omit the stores parameter to search all.'
    }

    const scopedNames = this._searchStores.map((s) => s.name)

    const inputSchema = z.object({
      query: z.string().describe('What to search for'),
      maxSearchResults: z.number().optional().describe('Maximum number of results per store'),
      stores: z
        .array(z.string())
        .optional()
        .describe('Filter to specific stores by name. Omit to search all available stores.'),
    })

    return tool({
      name: config.name ?? 'search_memory',
      description,
      inputSchema,
      callback: async (input) => {
        const stores = this._resolveToolTargets(scopedNames, input.stores)
        const results = await this.search(input.query, {
          ...(input.maxSearchResults != null && { maxSearchResults: input.maxSearchResults }),
          stores,
        })
        return results.map((entry) => ({
          content: entry.content,
          ...(entry.storeName && { storeName: entry.storeName }),
          ...(entry.metadata && { metadata: entry.metadata }),
        })) as JSONValue
      },
    })
  }

  private _createAddTool(config: MemoryToolConfig): Tool {
    let description = config.description ?? ADD_TOOL_DESCRIPTION
    const storeDescriptions = this._addStores.filter((s) => s.description).map((s) => `- ${s.name}: ${s.description}`)
    if (storeDescriptions.length > 0) {
      description += `\n\nAvailable writable stores:\n${storeDescriptions.join('\n')}`
      description +=
        '\n\nYou can target a specific store by name to route facts to the right place, or omit to add to all writable stores.'
    }

    const scopedNames = this._addStores.map((s) => s.name)

    const inputSchema = z.object({
      entries: z.array(z.string()).min(1).describe('Data to add to long-term memory'),
      stores: z
        .array(z.string())
        .optional()
        .describe('Target specific stores by name. Omit to add to all writable stores.'),
    })

    return tool({
      name: config.name ?? 'add_memory',
      description,
      inputSchema,
      callback: async (input) => {
        const stores = this._resolveToolTargets(scopedNames, input.stores)
        const settled = await Promise.allSettled(input.entries.map((content) => this.add(content, { stores })))
        const failures = settled.filter(
          (settledResult) => settledResult.status === 'rejected'
        ) as PromiseRejectedResult[]

        if (failures.length > 0) {
          const reasons = _flattenReasons(failures.map((failure) => failure.reason))
          throw new AggregateError(
            reasons,
            `MemoryManager: failed to add ${failures.length} of ${input.entries.length} entries: ${reasons.map((reason) => normalizeError(reason).message).join('; ')}`
          )
        }

        return (this._awaitWrites ? { stored: input.entries.length } : { accepted: input.entries.length }) as JSONValue
      },
    })
  }
}
