/**
 * Builder API for offload strategies.
 *
 * Offload strategies reduce content in L0 (the context window).
 * The builder composes a target, a reduction method (truncate, drop, or summarize),
 * and optional conditions into a strategy that implements `ContextStrategy`.
 *
 * @example
 * ```typescript
 * import { ContextManager, Offload } from '@strands-agents/sdk'
 *
 * const cm = new ContextManager({
 *   strategies: [
 *     Offload.truncate("toolResults", { previewTokens: 750 })
 *       .when({ threshold: 1500, skipRecent: 3 }),
 *     Offload.truncate(["bash", "list_files"], { previewTokens: 200 })
 *       .when({ threshold: 500 }),
 *     Offload.summarize({ ratio: 0.3 })
 *       .when({ utilization: 0.85 }),
 *     Offload("toolResultErrors"),  // drop from L0 entirely (still in L1)
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { ContextStrategy, StrategyContext, StrategyInitContext } from '../types.js'
import {
  estimateBlockTokens,
  isAlreadyTruncated,
  truncateBlock,
  type TruncateConfig,
} from './methods/truncate.js'
import type { SummarizeConfig } from './methods/summarize.js'
import {
  adjustSplitPointForToolPairs,
  generateSummary,
} from '../../conversation-manager/compression/context-compression.js'

const DEFAULT_THRESHOLD = 2500
const DEFAULT_SKIP_RECENT = 3
const DEFAULT_SUMMARIZE_RATIO = 0.3

/**
 * Target for offload operations.
 *
 * - `"toolResults"` — all successful tool result messages
 * - `"toolResultErrors"` — all failed tool result messages
 * - `"assistantMessages"` — assistant messages (text responses)
 * - `"userMessages"` — user messages (no tool result)
 * - `"images"` — messages containing image blocks
 * - `"documents"` — messages containing document blocks
 * - `string[]` — tool results from specific tools; prefix with `!` to exclude
 */
export type OffloadTarget =
  | 'toolResults'
  | 'toolResultErrors'
  | 'assistantMessages'
  | 'userMessages'
  | 'images'
  | 'documents'
  | string[]

/**
 * Conditions that determine when a strategy fires.
 */
export interface WhenConditions {
  /** Token threshold above which individual results are offloaded (truncate). */
  threshold?: number

  /** Context utilization ratio (0-1) above which the strategy fires (summarize). */
  utilization?: number

  /** Number of recent messages to skip during retroactive apply(). */
  skipRecent?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface StrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: WhenConditions): ContextStrategy
}


// --- Offload bare: drop from L0 entirely ---

class OffloadDropStrategy implements ContextStrategy {
  readonly name = 'offload:drop'

  private readonly _target: OffloadTarget
  private readonly _skipRecent: number
  private readonly _threshold: number
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, conditions?: WhenConditions) {
    this._target = target
    this._skipRecent = conditions?.skipRecent ?? DEFAULT_SKIP_RECENT
    this._threshold = conditions?.threshold ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      if (message.role !== 'user') return

      for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue

        if (this._threshold > 0) {
          const tokens = estimateBlockTokens(block)
          if (tokens <= this._threshold) continue
        }

        const replacement = new ToolResultBlock({
          toolUseId: block.toolUseId,
          status: block.status,
          content: [new TextBlock('[Dropped]')],
        })
        ;(message.content as unknown[])[blockIndex] = replacement
        logger.debug(`toolUseId=<${block.toolUseId}> | eagerly dropped tool result from L0`)
      }
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    const eligible = messages.slice(0, Math.max(0, messages.length - this._skipRecent))

    let dropped = false

    for (const message of eligible) {
      if (message.role !== 'user') continue

      for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue
        if (isAlreadyTruncated(block)) continue

        if (this._threshold > 0) {
          const tokens = estimateBlockTokens(block)
          if (tokens <= this._threshold) continue
        }

        const replacement = new ToolResultBlock({
          toolUseId: block.toolUseId,
          status: block.status,
          content: [new TextBlock('[Dropped]')],
        })
        ;(message.content as unknown[])[blockIndex] = replacement
        dropped = true
        logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from L0`)
      }
    }

    return dropped
  }

  private _matchesTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') return block.status !== 'error'
    if (this._target === 'toolResultErrors') return block.status === 'error'

    const toolName = resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) return !this._excludeFilter.has(toolName)
    if (this._toolFilter) return this._toolFilter.has(toolName)

    return true
  }
}

// --- Offload + Truncate strategy ---

class OffloadTruncateStrategy implements ContextStrategy {
  readonly name = 'offload:truncate'

  private readonly _threshold: number
  private readonly _skipRecent: number
  private readonly _truncateConfig: TruncateConfig
  private readonly _target: OffloadTarget
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, config?: TruncateConfig, conditions?: WhenConditions) {
    this._threshold = conditions?.threshold ?? DEFAULT_THRESHOLD
    this._skipRecent = conditions?.skipRecent ?? DEFAULT_SKIP_RECENT
    this._truncateConfig = config ?? {}
    this._target = target

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      if (message.role !== 'user') return

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue

        const estimatedTokens = estimateBlockTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const replacement = truncateBlock(block, this._truncateConfig)
        ;(message.content as unknown[])[blockIndex] = replacement
        logger.debug(
          `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | eagerly truncated tool result`
        )
      }
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    const eligible = messages.slice(0, Math.max(0, messages.length - this._skipRecent))

    let truncated = false

    for (const message of eligible) {
      if (message.role !== 'user') continue

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue
        if (isAlreadyTruncated(block)) continue

        const estimatedTokens = estimateBlockTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const replacement = truncateBlock(block, this._truncateConfig)
        ;(message.content as unknown[])[blockIndex] = replacement
        truncated = true
        logger.debug(
          `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | truncated tool result`
        )
      }
    }

    return truncated
  }

  private _matchesTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') return block.status !== 'error'
    if (this._target === 'toolResultErrors') return block.status === 'error'

    const toolName = resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) return !this._excludeFilter.has(toolName)
    if (this._toolFilter) return this._toolFilter.has(toolName)

    return true
  }
}

// --- Offload + Summarize strategy (conversation-level) ---

class OffloadSummarizeStrategy implements ContextStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig
  private readonly _utilization: number | undefined

  constructor(config?: SummarizeConfig, conditions?: WhenConditions) {
    this._config = config ?? {}
    this._utilization = conditions?.utilization
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization < this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping summarization, below threshold`
      )
      return false
    }

    const { messages, agent } = context
    const model = this._config.model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined

    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }

    const ratio = Math.max(0.1, Math.min(0.8, this._config.ratio ?? DEFAULT_SUMMARIZE_RATIO))

    let messagesToSummarize = Math.max(1, Math.floor(messages.length * ratio))

    if (messagesToSummarize >= messages.length) {
      logger.debug(`messages=<${messages.length}> | insufficient messages for summarization`)
      return false
    }

    try {
      messagesToSummarize = adjustSplitPointForToolPairs(messages, messagesToSummarize)
    } catch {
      logger.warn('unable to find valid split point for summarization')
      return false
    }

    const toSummarize = messages.slice(0, messagesToSummarize)
    if (toSummarize.length === 0) return false

    try {
      const summaryMessage = await generateSummary(toSummarize, model, this._config.systemPrompt)
      messages.splice(0, messagesToSummarize, summaryMessage)

      logger.debug(
        `summarized=<${messagesToSummarize}>, remaining=<${messages.length}> | summarized oldest messages`
      )
      return true
    } catch (error) {
      logger.warn(`error=<${error}> | conversation summarization failed`)
      return false
    }
  }
}

// --- Builder ---

function createDropBuilder(target: OffloadTarget): StrategyBuilder {
  const strategy = new OffloadDropStrategy(target)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadDropStrategy(target, conditions)
    },
  }
}

function createTruncateBuilder(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder {
  const strategy = new OffloadTruncateStrategy(target, config)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadTruncateStrategy(target, config, conditions)
    },
  }
}

function createSummarizeBuilder(config?: SummarizeConfig): StrategyBuilder {
  const strategy = new OffloadSummarizeStrategy(config)
  return {
    get name(): string {
      return strategy.name
    },
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadSummarizeStrategy(config, conditions)
    },
  }
}

/**
 * Offload strategy builder namespace.
 *
 * - `Offload(target)` — drop matching content from L0 entirely (still in L1)
 * - `Offload.truncate(target, config)` — replace with head-tail preview
 * - `Offload.summarize(config)` — summarize oldest messages (conversation-level)
 */
export interface OffloadNamespace {
  /** Drop matching content from L0 entirely (originals preserved in L1). */
  (target: OffloadTarget): StrategyBuilder

  /** Replace oversized content with a head-tail preview. */
  truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder

  /** Summarize the oldest messages to free context space (conversation-level). */
  summarize(config?: SummarizeConfig): StrategyBuilder
}

function offloadFn(target: OffloadTarget): StrategyBuilder {
  return createDropBuilder(target)
}

offloadFn.truncate = function truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder {
  return createTruncateBuilder(target, config)
}

offloadFn.summarize = function summarize(config?: SummarizeConfig): StrategyBuilder {
  return createSummarizeBuilder(config)
}

/**
 * Builder for offload strategies — reduces content in L0.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize({ ratio: 0.3 }).when({ utilization: 0.85 })
 * Offload("toolResultErrors")  // drop from L0 entirely
 * ```
 */
export const Offload: OffloadNamespace = offloadFn as OffloadNamespace

// --- Helpers ---

function resolveToolName(block: ToolResultBlock, message: Message): string | undefined {
  for (const content of message.content) {
    if ('toolUseId' in content && 'name' in content && (content as { toolUseId: string }).toolUseId === block.toolUseId) {
      return (content as { name: string }).name
    }
  }
  return undefined
}

function resolveToolFilter(target: OffloadTarget): { include?: Set<string>; exclude?: Set<string> } {
  if (typeof target === 'string') return {}
  if (!Array.isArray(target)) return {}

  const includes: string[] = []
  const excludes: string[] = []

  for (const entry of target) {
    if (entry.startsWith('!')) {
      excludes.push(entry.slice(1))
    } else {
      includes.push(entry)
    }
  }

  if (excludes.length > 0 && includes.length > 0) {
    return { include: new Set(includes) }
  }
  if (excludes.length > 0) {
    return { exclude: new Set(excludes) }
  }
  if (includes.length > 0) {
    return { include: new Set(includes) }
  }

  return {}
}
