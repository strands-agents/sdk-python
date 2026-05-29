import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Tool } from '../tools/tool.js'
import type {
  MemoryEntry,
  MemoryManagerConfig,
  MemorySearchOptions,
  MemoryStore,
  MemoryStoreOptions,
  MemoryToolConfig,
} from './types.js'
import type { JSONValue } from '../types/json.js'
import { tool } from '../tools/tool-factory.js'
import { z } from 'zod'
import { logger } from '../logging/logger.js'

const SEARCH_TOOL_DESCRIPTION =
  'Search long-term memory for facts, preferences, or context from previous conversations. Use when you need background about the user or topic that may have been discussed before.'

const STORE_TOOL_DESCRIPTION =
  'Store facts, preferences, or decisions that should be remembered across conversations. Use when the user shares something worth recalling later.'

const DEFAULT_RESULTS_PER_STORE = 3

/**
 * Provides cross-session knowledge retrieval and storage for agents.
 *
 * Manages one or more {@link MemoryStore} backends, exposing `search_memory` and
 * `store_memory` tools for agent-driven recall and persistence.
 *
 * @example
 * ```typescript
 * import { Agent, MemoryManager } from '@strands-agents/sdk'
 *
 * // Config shorthand
 * const agent = new Agent({
 *   model,
 *   memoryManager: { stores: [myStore], storeToolConfig: true },
 * })
 *
 * // Class instance (for programmatic access)
 * const memoryManager = new MemoryManager({ stores: [myStore], storeToolConfig: true })
 * const agent = new Agent({ model, memoryManager })
 * await memoryManager.search('user preferences')
 * ```
 */
export class MemoryManager implements Plugin {
  readonly name = 'strands:memory-manager'
  private readonly _config: MemoryManagerConfig
  private readonly _searchStores: MemoryStore[]
  private readonly _storeStores: MemoryStore[]
  private readonly _searchToolConfig: MemoryToolConfig | false
  private readonly _storeToolConfig: MemoryToolConfig | false

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
    }

    this._config = config

    if (config.searchToolConfig === false) {
      this._searchToolConfig = false
      this._searchStores = []
    } else {
      const toolConfig = typeof config.searchToolConfig === 'object' ? config.searchToolConfig : {}
      this._searchStores = this._resolveStores(config.stores, toolConfig.stores)
      this._searchToolConfig = toolConfig
    }

    if (config.storeToolConfig === undefined || config.storeToolConfig === false) {
      this._storeToolConfig = false
      this._storeStores = []
    } else {
      const toolConfig = typeof config.storeToolConfig === 'object' ? config.storeToolConfig : {}
      const resolved = this._resolveStores(config.stores, toolConfig.stores).filter((s) => s.add)

      if (resolved.length === 0) {
        throw new Error('MemoryManager: storeToolConfig targets no writable stores')
      }

      if (config.storeToolConfig === true && resolved.length > 1 && !toolConfig.stores) {
        throw new Error(
          'MemoryManager: storeToolConfig must specify `stores` when multiple writable stores are configured'
        )
      }

      this._storeStores = resolved
      this._storeToolConfig = toolConfig
    }
  }

  /**
   * Initializes the plugin with the agent.
   *
   * No lifecycle hooks are registered in this version; context injection and extraction
   * triggers are deferred to a follow-up PR. Tool registration is handled automatically
   * by the PluginRegistry via {@link getTools}.
   *
   * @param _agent - The agent this plugin is being attached to
   */
  initAgent(_agent: LocalAgent): void {}

  /**
   * Returns tools registered by this plugin.
   *
   * @returns Array of tools to register with the agent
   */
  getTools(): Tool[] {
    const tools: Tool[] = []

    if (this._searchToolConfig !== false) {
      tools.push(this._createSearchTool(this._searchToolConfig))
    }

    if (this._storeToolConfig !== false) {
      tools.push(this._createStoreTool(this._storeToolConfig))
    }

    return tools
  }

  /**
   * Search configured stores for entries matching the query.
   *
   * This method is intentionally unscoped (full access to all configured stores); it is the
   * programmatic escape hatch. Tool-level store scoping is applied by the search tool callback.
   * When `options.stores` is omitted, all stores are searched; an empty array searches none.
   *
   * Each store receives the `limit` individually — results are concatenated in store config order.
   * Stores that fail are logged and skipped.
   *
   * @param query - The search query string
   * @param options - Optional limit per-store and store name filter
   * @returns Array of memory entries from matching stores
   */
  async search(query: string, options?: MemorySearchOptions): Promise<MemoryEntry[]> {
    logger.debug(`query=<${query}>, limit=<${options?.limit}>, stores=<${options?.stores}> | searching stores`)

    const targetStores =
      options?.stores !== undefined
        ? this._config.stores.filter((s) => options.stores!.includes(s.name))
        : this._config.stores

    if (options?.stores !== undefined && targetStores.length === 0) {
      logger.warn(`stores=<${options.stores.join(', ')}> | no stores matched filter`)
    }

    const limit = options?.limit
    const settled = await Promise.allSettled(
      targetStores.map((store) => store.search(query, { limit: limit ?? store.limit ?? DEFAULT_RESULTS_PER_STORE }))
    )

    const results: MemoryEntry[] = []
    for (let i = 0; i < settled.length; i++) {
      const r = settled[i]!
      if (r.status === 'rejected') {
        logger.warn(`store=<${targetStores[i]!.name}>, reason=<${r.reason}> | store search failed`)
        continue
      }
      for (const entry of r.value) {
        results.push(entry)
      }
    }

    logger.debug(`results=<${results.length}> | search complete`)
    return results
  }

  /**
   * Store content in writable stores. If `stores` is provided, only writes to those named stores.
   *
   * This method is intentionally unscoped (full access to all configured writable stores); it is
   * the programmatic escape hatch. Tool-level store scoping is applied by the store tool callback.
   * When `options.stores` is omitted, all writable stores are targeted; an empty array targets none.
   *
   * Partial failures are logged. If all writes fail, throws an `AggregateError`.
   *
   * @param content - The text content to store
   * @param options - Optional metadata and store name filter
   */
  async store(content: string, options?: MemoryStoreOptions): Promise<void> {
    let writableStores: MemoryStore[]

    if (options?.stores !== undefined) {
      writableStores = options.stores.map((name) => {
        const found = this._config.stores.find((s) => s.name === name)
        if (!found) {
          throw new Error(`MemoryManager: store '${name}' not found`)
        }
        if (!found.add) {
          throw new Error(`MemoryManager: store '${name}' is read-only`)
        }
        return found
      })
    } else {
      writableStores = this._config.stores.filter((s) => s.add)
    }

    if (writableStores.length === 0) {
      throw new Error('MemoryManager: no writable store matched')
    }

    const settled = await Promise.allSettled(writableStores.map((s) => s.add!(content, options?.metadata)))

    const failures: { store: string; reason: unknown }[] = []
    for (let i = 0; i < settled.length; i++) {
      const r = settled[i]!
      if (r.status === 'rejected') {
        const storeName = writableStores[i]!.name
        logger.warn(`store=<${storeName}>, reason=<${r.reason}> | store write failed`)
        failures.push({ store: storeName, reason: r.reason })
      }
    }
    if (failures.length === writableStores.length) {
      throw new AggregateError(
        failures.map((f) => f.reason),
        `MemoryManager: all store writes failed: ${failures.map((f) => f.store).join(', ')}`
      )
    }
  }

  private _resolveStores(allStores: MemoryStore[], scoped?: string[]): MemoryStore[] {
    if (!scoped || scoped.length === 0) return allStores

    return scoped.map((name) => {
      const found = allStores.find((s) => s.name === name)
      if (!found) {
        throw new Error(`MemoryManager: store '${name}' not found`)
      }
      return found
    })
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
      limit: z.number().optional().describe('Maximum number of results per store'),
      stores: z
        .array(z.string())
        .optional()
        .describe(
          'Filter to specific stores by name. Omit entirely to search all available stores (do not pass an empty array).'
        ),
    })

    return tool({
      name: config.name ?? 'search_memory',
      description,
      inputSchema,
      callback: async (input) => {
        const stores = input.stores != null ? input.stores.filter((name) => scopedNames.includes(name)) : scopedNames
        const results = await this.search(input.query, {
          ...(input.limit != null && { limit: input.limit }),
          stores,
        })
        return results.map((entry) => ({
          content: entry.content,
          ...(entry.metadata && { metadata: entry.metadata }),
        })) as JSONValue
      },
    })
  }

  private _createStoreTool(config: MemoryToolConfig): Tool {
    let description = config.description ?? STORE_TOOL_DESCRIPTION
    const storeDescriptions = this._storeStores.filter((s) => s.description).map((s) => `- ${s.name}: ${s.description}`)
    if (storeDescriptions.length > 0) {
      description += `\n\nAvailable writable stores:\n${storeDescriptions.join('\n')}`
      description +=
        '\n\nYou can target a specific store by name to route facts to the right place, or omit to store in all writable stores.'
    }

    const scopedNames = this._storeStores.map((s) => s.name)

    const inputSchema = z.object({
      entries: z.array(z.string()).describe('Data to store in long-term memory'),
      stores: z
        .array(z.string())
        .optional()
        .describe(
          'Target specific stores by name. Omit entirely to store in all writable stores (do not pass an empty array).'
        ),
    })

    return tool({
      name: config.name ?? 'store_memory',
      description,
      inputSchema,
      callback: async (input) => {
        const stores = input.stores != null ? input.stores.filter((name) => scopedNames.includes(name)) : scopedNames
        const settled = await Promise.allSettled(input.entries.map((content) => this.store(content, { stores })))
        const stored = settled.filter((r) => r.status === 'fulfilled').length
        const failed = settled.filter((r) => r.status === 'rejected').length
        return { stored, failed } as JSONValue
      },
    })
  }
}
