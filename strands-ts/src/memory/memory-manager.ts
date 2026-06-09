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
  MemoryAddToolConfig,
} from './types.js'
import type { JSONValue } from '../types/json.js'
import { MessageAddedEvent } from '../hooks/events.js'
import { ExtractionCoordinator } from './extraction/coordinator.js'
import type { ExtractionTrigger } from './extraction/types.js'
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
 * Whether a store has any write sink. The `add_memory` tool and programmatic `add` use `add`;
 * extraction additionally accepts `addMessages`. A writable store must have at least one.
 */
function _hasWriteSink(store: MemoryStore): boolean {
  return typeof store.add === 'function' || typeof store.addMessages === 'function'
}

/** Normalizes a store's `trigger` field (a single trigger or an array) to an array. */
function _normalizeTriggers(trigger: ExtractionTrigger | ExtractionTrigger[]): ExtractionTrigger[] {
  return Array.isArray(trigger) ? trigger : [trigger]
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
  /** All writable stores — the unscoped target set for the programmatic {@link add} method. */
  private readonly _addStores: MemoryStore[]
  private readonly _searchToolConfig: MemoryToolConfig | false
  private readonly _addToolConfig: MemoryAddToolConfig | false
  private readonly _addToolStores: MemoryStore[]
  /** Stores with an extraction config and at least one trigger; wired up in {@link initAgent}. */
  private readonly _extractionStores: MemoryStore[]
  /** Background extraction coordinator, created in {@link initAgent} when extraction is configured. */
  private _coordinator: ExtractionCoordinator | undefined

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

      if (store.writable && !_hasWriteSink(store)) {
        throw new Error(`MemoryManager: store '${store.name}' is writable but has no add or addMessages method`)
      }

      if (store.extraction) {
        if (!store.writable) {
          throw new Error(`MemoryManager: store '${store.name}' has extraction config but is not writable`)
        }
        if (_normalizeTriggers(store.extraction.trigger).length === 0) {
          throw new Error(`MemoryManager: store '${store.name}' has extraction config but no triggers`)
        }
        // Each extraction shape needs its matching write sink. An extractor produces discrete entries
        // written via `add`; without an extractor the raw message batch goes to `addMessages`.
        if (store.extraction.extractor) {
          if (typeof store.add !== 'function') {
            throw new Error(
              `MemoryManager: store '${store.name}' has an extractor but no add method (extracted entries are written via add)`
            )
          }
        } else if (typeof store.addMessages !== 'function') {
          throw new Error(
            `MemoryManager: store '${store.name}' has extraction config without an extractor but no addMessages method`
          )
        }
      }
    }

    this._config = config
    this._searchStores = config.stores
    // `add`-targeting paths (tool / programmatic) need an `add` method specifically.
    this._addStores = config.stores.filter((s) => s.writable && typeof s.add === 'function')
    this._extractionStores = config.stores.filter((s) => s.writable && s.extraction)

    this._searchToolConfig =
      config.searchToolConfig === false
        ? false
        : typeof config.searchToolConfig === 'object'
          ? config.searchToolConfig
          : {}

    if (config.addToolConfig === undefined || config.addToolConfig === false) {
      this._addToolConfig = false
      this._addToolStores = []
    } else {
      // The `add_memory` tool writes via `add` (not `addMessages`), so it needs an `add`-capable store.
      if (this._addStores.length === 0) {
        throw new Error('MemoryManager: addToolConfig is enabled but no writable stores implement add')
      }
      this._addToolConfig = typeof config.addToolConfig === 'object' ? config.addToolConfig : {}
      this._addToolStores = this._resolveAddToolStores(this._addToolConfig)
    }
  }

  /**
   * Resolves the writable stores the `add_memory` tool may write to. When `stores` is given, each
   * entry (a store name or a {@link MemoryStore} instance) must resolve by name to a configured,
   * `add`-capable writable store (else throws). Omitted means all such stores.
   */
  private _resolveAddToolStores(toolConfig: MemoryAddToolConfig): MemoryStore[] {
    if (toolConfig.stores === undefined) {
      return this._addStores
    }

    const names = toolConfig.stores.map((store) => (typeof store === 'string' ? store : store.name))

    return [...new Set(names)].map((name) => {
      const found = this._config.stores.find((s) => s.name === name)
      if (!found) {
        throw new Error(`MemoryManager: addToolConfig store '${name}' not found`)
      }
      if (!found.writable) {
        throw new Error(`MemoryManager: addToolConfig store '${name}' is not writable`)
      }
      if (typeof found.add !== 'function') {
        throw new Error(`MemoryManager: addToolConfig store '${name}' has no add method (only addMessages)`)
      }
      return found
    })
  }

  /**
   * Initializes the plugin with the agent.
   *
   * Wires up automatic extraction for any store configured with {@link ExtractionConfig}: buffers
   * conversation messages and attaches each store's triggers. A no-op when no store uses extraction.
   *
   * @param agent - The agent this plugin is being attached to
   */
  initAgent(agent: LocalAgent): void {
    if (this._extractionStores.length === 0) {
      return
    }

    const coordinator = new ExtractionCoordinator(this._extractionStores, agent.model)
    this._coordinator = coordinator

    // Buffer every message the agent adds, so extraction has its own copy to save from.
    agent.addHook(MessageAddedEvent, (event) => {
      coordinator.record(event.message.toJSON())
    })

    for (const store of this._extractionStores) {
      for (const trigger of _normalizeTriggers(store.extraction!.trigger)) {
        trigger.attach({ agent, fire: () => void coordinator.process(store) })
      }
    }
  }

  /**
   * Saves every store's remaining messages and waits for all saves to finish. No-op when no store has
   * extraction configured.
   *
   * Extraction normally runs in the background, so the most recent turn may not be saved yet when the
   * agent responds. Call this once at a boundary you control - typically your app's shutdown handler -
   * so nothing is lost. A process killed before then (crash, hard timeout) may still lose the last
   * unsaved turn; a more frequent trigger narrows that window.
   *
   * Do not call this after every turn alongside a periodic trigger: it forces a save each time and so
   * defeats the trigger's schedule.
   */
  async flush(): Promise<void> {
    await this._coordinator?.flush()
  }

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
      tools.push(this._createAddTool(this._addToolConfig, this._addToolStores))
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
        results.push({ ...entry, storeName })
      }
    }

    logger.debug(`results=<${results.length}> | search complete`)
    return results
  }

  /**
   * Add content to writable stores. If `stores` is provided, only writes to those named stores;
   * otherwise all writable stores are targeted.
   *
   * This method is unscoped, with full access to all configured writable stores; tool-level store
   * scoping is applied by the add tool callback. Target stores are validated first (an unknown or
   * read-only named store throws), then the writes are awaited: per-store failures are logged, and
   * an `AggregateError` is thrown if any store fails.
   *
   * @param content - The text content to add
   * @param options - Optional metadata and store name filter
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

    const settled = await Promise.allSettled(writableStores.map((store) => store.add!(content, options?.metadata)))

    const failures: { store: string; reason: unknown }[] = []
    for (let i = 0; i < settled.length; i++) {
      const settledResult = settled[i]!
      if (settledResult.status === 'rejected') {
        const storeName = writableStores[i]!.name
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

  private _createAddTool(config: MemoryAddToolConfig, stores: MemoryStore[]): Tool {
    let description = config.description ?? ADD_TOOL_DESCRIPTION
    const storeDescriptions = stores.filter((s) => s.description).map((s) => `- ${s.name}: ${s.description}`)
    if (storeDescriptions.length > 0) {
      description += `\n\nAvailable writable stores:\n${storeDescriptions.join('\n')}`
      description +=
        '\n\nYou can target a specific store by name to route facts to the right place, or omit to add to all available writable stores.'
    }

    const scopedNames = stores.map((s) => s.name)
    const waitForWrites = config.waitForWrites ?? true

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

        if (!waitForWrites) {
          // Fire-and-forget: dispatch the writes without awaiting so the agent loop isn't blocked.
          // add() logs per-store failures; swallow the rejection so it isn't an unhandled rejection.
          for (const content of input.entries) {
            void this.add(content, { stores }).catch(() => {})
          }
          return { accepted: input.entries.length } as JSONValue
        }

        // Await mode: surface failures to the model with concrete reasons (not nested AggregateErrors).
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

        return { stored: input.entries.length } as JSONValue
      },
    })
  }
}
