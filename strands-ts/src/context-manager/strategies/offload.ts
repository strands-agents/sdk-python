/**
 * Builder API for offload strategies.
 *
 * Offload strategies move content out of L0 (the context window) and into storage.
 * The builder composes a target, a reduction method (truncate or summarize), and
 * optional conditions into a strategy that implements `ContextStrategy`.
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
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { ToolResultBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { ContextStrategy, StrategyContext, StrategyInitContext } from '../types.js'
import {
  estimateBlockTokens,
  extractBlockText,
  isAlreadyTruncated,
  truncateBlock,
  type TruncateConfig,
} from './methods/truncate.js'
import { summarizeMessages, type SummarizeConfig } from './methods/summarize.js'

const DEFAULT_THRESHOLD = 2500
const DEFAULT_SKIP_RECENT = 3

/**
 * Target for offload operations.
 *
 * - `"toolResults"` — all successful tool results
 * - `"toolResultErrors"` — all error tool results
 * - `string[]` — specific tool names to include; prefix with `!` to exclude
 */
export type OffloadTarget = 'toolResults' | 'toolResultErrors' | string[]

/**
 * Conditions that determine when a strategy fires.
 */
export interface WhenConditions {
  /** Token threshold above which individual results are offloaded (truncate only). */
  threshold?: number

  /** Context utilization ratio (0-1) above which the strategy fires (summarize only). */
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
    const { agent, storage } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      if (message.role !== 'user') return

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue

        const estimatedTokens = estimateBlockTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const storageKey = `offload/${block.toolUseId}`
        const fullText = extractBlockText(block)

        try {
          await storage.write(storageKey, new TextEncoder().encode(fullText))
        } catch {
          logger.warn(`toolUseId=<${block.toolUseId}> | failed to store offloaded content`)
          continue
        }

        const replacement = truncateBlock(block, storageKey, this._truncateConfig)
        ;(message.content as unknown[])[blockIndex] = replacement
        logger.debug(
          `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | eagerly offloaded tool result to storage`
        )
      }
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages, storage } = context
    const eligible = messages.slice(0, Math.max(0, messages.length - this._skipRecent))

    let offloaded = false

    for (const message of eligible) {
      if (message.role !== 'user') continue

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue
        if (isAlreadyTruncated(block)) continue

        const estimatedTokens = estimateBlockTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const storageKey = `offload/${block.toolUseId}`
        const fullText = extractBlockText(block)

        try {
          await storage.write(storageKey, new TextEncoder().encode(fullText))
        } catch {
          logger.warn(`toolUseId=<${block.toolUseId}> | failed to store offloaded content`)
          continue
        }

        const replacement = truncateBlock(block, storageKey, this._truncateConfig)
        ;(message.content as unknown[])[blockIndex] = replacement
        offloaded = true
        logger.debug(
          `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | offloaded tool result to storage`
        )
      }
    }

    return offloaded
  }

  private _matchesTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') {
      return block.status !== 'error'
    }
    if (this._target === 'toolResultErrors') {
      return block.status === 'error'
    }

    const toolName = this._resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) {
      return !this._excludeFilter.has(toolName)
    }
    if (this._toolFilter) {
      return this._toolFilter.has(toolName)
    }

    return true
  }

  private _resolveToolName(block: ToolResultBlock, message: Message): string | undefined {
    for (const content of message.content) {
      if ('toolUseId' in content && 'name' in content && (content as { toolUseId: string }).toolUseId === block.toolUseId) {
        return (content as { name: string }).name
      }
    }
    return undefined
  }
}

// --- Offload + Summarize strategy ---

class OffloadSummarizeStrategy implements ContextStrategy {
  readonly name = 'offload:summarize'

  private readonly _summarizeConfig: SummarizeConfig
  private readonly _utilization: number | undefined

  constructor(config?: SummarizeConfig, conditions?: WhenConditions) {
    this._summarizeConfig = config ?? {}
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
    const model = this._summarizeConfig.model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined

    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }

    return summarizeMessages(messages, model, this._summarizeConfig)
  }
}

// --- Builder ---

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
 * Use as a namespace with static methods for constructing offload strategies:
 * - `Offload.truncate(target, config)` — truncate oversized content with a preview
 * - `Offload.summarize(config)` — summarize oldest messages
 *
 * Or call directly as a function for simple target-only offloading:
 * - `Offload("toolResults")` — offload all tool results with default settings
 */
export interface OffloadNamespace {
  /** Shorthand: offload with default truncate method for the given target. */
  (target: OffloadTarget): StrategyBuilder

  /** Create a strategy that truncates oversized content into a head-tail preview. */
  truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder

  /** Create a strategy that summarizes the oldest messages to free token space. */
  summarize(config?: SummarizeConfig): StrategyBuilder
}

function offloadFn(target: OffloadTarget): StrategyBuilder {
  return createTruncateBuilder(target)
}

offloadFn.truncate = function truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder {
  return createTruncateBuilder(target, config)
}

offloadFn.summarize = function summarize(config?: SummarizeConfig): StrategyBuilder {
  return createSummarizeBuilder(config)
}

/**
 * Builder for offload strategies — moves content out of L0 into storage.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize({ ratio: 0.3 }).when({ utilization: 0.85 })
 * Offload("toolResultErrors")
 * ```
 */
export const Offload: OffloadNamespace = offloadFn as OffloadNamespace

// --- Helpers ---

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
