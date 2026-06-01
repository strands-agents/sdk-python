import type { Storage } from '../vended-plugins/context-offloader/storage.js'
import type { Plugin } from '../plugins/plugin.js'
import type { Tool } from '../tools/tool.js'
import type { LocalAgent } from '../types/agent.js'
import { ContextCompression } from './compression/context-compression.js'
import { ContextOffloader } from '../vended-plugins/context-offloader/plugin.js'
import { InMemoryStorage } from '../vended-plugins/context-offloader/storage.js'

export type ContextStrategyValue = 'auto'

/**
 * Configuration for the offloader component.
 */
export type OffloaderConfig = {
  /** Token threshold above which tool results are offloaded. Defaults to 2500. */
  threshold?: number
  /** Number of tokens to keep as an inline preview. Defaults to 1500. */
  previewTokens?: number
}

/**
 * Compression configuration accepted by contextManager.
 * - `true`: enable with defaults (truncate, proactive at 0.7).
 * - `'truncate'` / `'summarize'`: enable specific strategy with defaults.
 * - Object: full config with strategy and options.
 * - Omitted: disabled.
 */
export type CompressionConfig =
  | true
  | import('./compression/context-compression.js').CompressionMethod
  | import('./compression/context-compression.js').CompressionOptions

/**
 * Configuration accepted by the {@link ContextManager} constructor.
 *
 * Config objects are additive — only features you explicitly set are enabled.
 * Use `"auto"` to enable everything with defaults.
 */
export type ContextManagerConfig = {
  /** Strategy name. Only "auto" is supported currently. */
  strategy?: ContextStrategyValue
  /** Storage backend for cached tool results. Defaults to InMemoryStorage. */
  storage?: Storage
  /**
   * Context offloader configuration.
   * - `true`: enable with defaults (threshold=2500, previewTokens=500).
   * - Object: enable with custom settings.
   * - Omitted: disabled.
   */
  offloader?: true | OffloaderConfig
  /**
   * Compression configuration.
   * - `true`: enable with defaults (truncate, proactive at 0.7).
   * - `'truncate'` / `'summarize'`: enable specific strategy with defaults.
   * - `CompressionStrategy.Truncate(...)` / `CompressionStrategy.Summarize(...)`: full config.
   * - Omitted: disabled.
   */
  compression?: CompressionConfig
}

/**
 * The `contextManager` parameter type accepted by AgentConfig.
 *
 * - `"auto"`: enables everything with defaults.
 * - `{ strategy: 'auto', ... }`: auto with overrides (omitted features stay enabled).
 * - `{ compression: true }`: additive — only what you set is enabled.
 * - `undefined` (default): no context management facade.
 */
export type ContextManagerParam = ContextStrategyValue | ContextManagerConfig

/**
 * Pre-composed context management for agents.
 *
 * Internal plugin that composes sub-plugins (ContextCompression, ContextOffloader)
 * for compression and caching behavior.
 *
 * @example
 * ```typescript
 * const agent = new Agent({ contextManager: "auto" })
 * ```
 */
export class ContextManager implements Plugin {
  readonly name = 'strands:context-manager'
  readonly storage: Storage

  private readonly _config: ContextManagerConfig
  private _subPlugins: Plugin[] | undefined

  constructor(config?: ContextManagerConfig) {
    this._config = config ?? {}
    this.storage = this._config.storage ?? new InMemoryStorage()
  }

  /**
   * Resolve sub-plugins, skipping any that the user already provides.
   * Called once before plugin initialization.
   * @internal
   */
  _resolveSubPlugins(userPlugins?: Plugin[]): void {
    this._subPlugins = this._buildSubPlugins(userPlugins)
  }

  getTools(): Tool[] {
    const plugins = this._subPlugins ?? []
    const tools: Tool[] = []
    for (const plugin of plugins) {
      if (plugin.getTools) {
        tools.push(...plugin.getTools())
      }
    }
    return tools
  }

  initAgent(agent: LocalAgent): void {
    if (!this._subPlugins) {
      this._subPlugins = this._buildSubPlugins()
    }

    for (const plugin of this._subPlugins) {
      plugin.initAgent(agent)
    }
  }

  private _buildSubPlugins(userPlugins?: Plugin[]): Plugin[] {
    const config = this._config
    const plugins: Plugin[] = []

    if (config.compression) {
      const userProvided = userPlugins?.some((p) => p.name === 'strands:context-compression')
      if (!userProvided) {
        let compressionConfig: import('./compression/context-compression.js').CompressionOptions | undefined
        if (config.compression === true) {
          compressionConfig = undefined
        } else if (typeof config.compression === 'string') {
          compressionConfig = { method: config.compression }
        } else {
          compressionConfig = config.compression
        }
        plugins.push(new ContextCompression(compressionConfig))
      }
    }

    if (config.offloader) {
      const userProvided = userPlugins?.some((p) => p.name === 'strands:context-offloader')
      if (!userProvided) {
        const offloaderConfig = config.offloader === true ? {} : config.offloader
        plugins.push(
          new ContextOffloader({
            storage: this.storage,
            maxResultTokens: offloaderConfig.threshold ?? 2500,
            previewTokens: offloaderConfig.previewTokens ?? 1500,
            includeRetrievalTool: true,
          })
        )
      }
    }

    return plugins
  }
}

/**
 * Resolve a `contextManager` parameter into a ContextManager plugin instance.
 * User-provided plugins that overlap with sub-plugins take precedence.
 *
 * @param param - The contextManager config (strategy string, config object, or class instance)
 * @param userPlugins - User-provided plugins array, used for dedup checking
 * @internal
 */
const STRATEGY_DEFAULTS = {
  auto: { compression: true, offloader: true },
} satisfies Record<string, Partial<ContextManagerConfig>>

export function resolveContextManager(param: ContextManagerParam, userPlugins?: Plugin[]): ContextManager {
  const base = typeof param === 'string' ? { strategy: param } : param
  const defaults = base.strategy ? STRATEGY_DEFAULTS[base.strategy] : undefined
  const config = defaults ? { ...defaults, ...base } : base

  const instance = new ContextManager(config)
  instance._resolveSubPlugins(userPlugins)
  return instance
}
