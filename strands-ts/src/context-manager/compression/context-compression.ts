import type { Plugin } from '../../plugins/plugin.js'
import type { LocalAgent } from '../../types/agent.js'
import type { Tool } from '../../tools/tool.js'
import type { Message } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../../hooks/events.js'
import { ContextWindowOverflowError } from '../../errors.js'
import { truncate, type TruncateOptions } from './strategies/truncate.js'
import { summarize, type SummarizeOptions } from './strategies/summarize.js'
import { estimateInputTokens } from '../token-estimation.js'
import { logger } from '../../logging/logger.js'
import { warnOnce } from '../../logging/warn-once.js'

const DEFAULT_CONTEXT_WINDOW_LIMIT = 200_000
const DEFAULT_PROACTIVE_THRESHOLD = 0.7
const DEFAULT_WINDOW_SIZE = 40

export type CompressionMethod = 'truncate' | 'summarize'

type SharedCompressionOptions = {
  /**
   * Proactive compression before the model call.
   * - `true`: compress when 70% of the context window is used (default threshold).
   * - `{ threshold: number }`: compress at the specified ratio (0, 1].
   * - `false`: disable proactive compression; only reactive overflow recovery is used.
   * - Omitted: defaults to `true`.
   */
  proactive?: boolean | { threshold: number }
  /**
   * Number of messages at the start of the conversation to protect from eviction.
   * These messages are never trimmed or summarized away.
   */
  protectFirst?: number
}

export type TruncateCompressionConfig = SharedCompressionOptions & {
  method?: 'truncate'
  /** Maximum messages to keep after trimming. Defaults to 40. */
  windowSize?: number
}

export type SummarizeCompressionConfig = SharedCompressionOptions & {
  method: 'summarize'
  /** Ratio of messages to summarize (0.1–0.8). Defaults to 0.3. */
  summaryRatio?: number
  /** Minimum recent messages to preserve during summarization. Defaults to 10. */
  preserveRecentMessages?: number
}

/**
 * Compression configuration (discriminated union on `method`).
 *
 * @example
 * ```typescript
 * contextManager: { compression: true }                                         // defaults (truncate)
 * contextManager: { compression: 'summarize' }                                  // strategy shorthand
 * contextManager: { compression: { method: 'truncate', windowSize: 30 } }     // full config
 * contextManager: { compression: { method: 'summarize', summaryRatio: 0.5 } } // full config
 * ```
 */
export type CompressionOptions = TruncateCompressionConfig | SummarizeCompressionConfig

/**
 * Plugin that handles context compression — both proactive (before model call when
 * threshold is exceeded) and reactive (after model call on overflow error).
 *
 * Delegates reduction to strategy functions (truncate or summarize).
 */
export class ContextCompression implements Plugin {
  readonly name = 'strands:context-compression'

  private readonly _proactiveThreshold: number | undefined
  private readonly _method: CompressionMethod
  private readonly _windowSize: number
  private readonly _truncateOptions: TruncateOptions
  private readonly _summarizeOptions: SummarizeOptions

  constructor(config?: CompressionOptions) {
    const proactive = config?.proactive ?? true
    if (proactive === false) {
      this._proactiveThreshold = undefined
    } else if (proactive === true) {
      this._proactiveThreshold = DEFAULT_PROACTIVE_THRESHOLD
    } else {
      if (proactive.threshold <= 0 || proactive.threshold > 1) {
        throw new Error(
          `proactive compression threshold must be between 0 (exclusive) and 1 (inclusive), got ${proactive.threshold}`
        )
      }
      this._proactiveThreshold = proactive.threshold
    }

    this._method = config?.method ?? 'truncate'
    this._windowSize = (config as TruncateCompressionConfig | undefined)?.windowSize ?? DEFAULT_WINDOW_SIZE

    this._truncateOptions = {
      ...(config?.protectFirst !== undefined && { protectFirst: config.protectFirst }),
    }
    this._summarizeOptions = {
      ...(config?.protectFirst !== undefined && { protectFirst: config.protectFirst }),
      ...((config as SummarizeCompressionConfig)?.summaryRatio !== undefined && {
        summaryRatio: (config as SummarizeCompressionConfig).summaryRatio,
      }),
      ...((config as SummarizeCompressionConfig)?.preserveRecentMessages !== undefined && {
        preserveRecentMessages: (config as SummarizeCompressionConfig).preserveRecentMessages,
      }),
    }
  }

  getTools(): Tool[] {
    return []
  }

  initAgent(agent: LocalAgent): void {
    // Reactive overflow recovery
    agent.addHook(AfterModelCallEvent, async (event) => {
      if (event.error instanceof ContextWindowOverflowError) {
        if (await this._reduce(event.agent.messages, event.model)) {
          event.retry = true
        }
      }
    })

    // Proactive compression
    agent.addHook(BeforeModelCallEvent, async (event) => {
      if (this._proactiveThreshold === undefined) {
        return
      }

      let contextWindowLimit = event.model.getConfig().contextWindowLimit
      if (contextWindowLimit === undefined) {
        contextWindowLimit = DEFAULT_CONTEXT_WINDOW_LIMIT
        warnOnce(
          logger,
          `context_compression | contextWindowLimit is not set on the model, using default of ${DEFAULT_CONTEXT_WINDOW_LIMIT} | set contextWindowLimit in your model config for accurate proactive compression`
        )
      }

      const projectedInputTokens =
        event.projectedInputTokens ?? (await estimateInputTokens(event.agent.messages, event.model))

      if (projectedInputTokens === undefined) {
        return
      }

      const ratio = projectedInputTokens / contextWindowLimit
      if (ratio >= this._proactiveThreshold) {
        logger.debug(
          `projected_tokens=<${projectedInputTokens}>, limit=<${contextWindowLimit}>, ratio=<${ratio.toFixed(2)}>, threshold=<${this._proactiveThreshold}> | compression threshold exceeded, reducing context`
        )
        try {
          await this._reduce(event.agent.messages, event.model)
        } catch (e) {
          logger.warn(`context_compression | proactive compression failed, continuing | error=<${e}>`)
        }
      }
    })

    // Sliding window enforcement after each invocation (truncate method only)
    if (this._method === 'truncate') {
      agent.addHook(AfterInvocationEvent, (event) => {
        if (event.agent.messages.length > this._windowSize) {
          truncate(event.agent.messages, this._windowSize, this._truncateOptions)
        }
      })
    }
  }

  private async _reduce(messages: Message[], model: Model): Promise<boolean> {
    switch (this._method) {
      case 'summarize':
        return summarize(messages, model, this._summarizeOptions)
      case 'truncate':
      default:
        return truncate(messages, this._windowSize, this._truncateOptions)
    }
  }
}
