/**
 * Builder API for offload strategies.
 *
 * Offload strategies reduce content in the context window.
 * The builder composes a target, a reduction method (truncate, drop, or summarize),
 * and optional conditions into a strategy that implements `ContextStrategy`.
 *
 * Conditions determine granularity:
 * - `threshold` → per-block (act on individual blocks above this size)
 * - `utilization` without `threshold` → message-level batch (sliding window or batched summary)
 * - Both → per-block, gated by utilization
 *
 * @example
 * ```typescript
 * import { ContextManager, Offload } from '@strands-agents/sdk/context-manager'
 *
 * const cm = new ContextManager({
 *   strategies: [
 *     // Per-block: truncate individual tool results over 2500 tokens, eagerly on arrival
 *     Offload.truncate("toolResults", { previewTokens: 750 })
 *       .when({ threshold: 1500 }),
 *
 *     // Per-block: truncate specific tools
 *     Offload.truncate(["tool::bash", "tool::read_file"]).when({ threshold: 2000 }),
 *
 *     // Message-level: one LLM call summarizing oldest messages on overflow
 *     Offload.summarize("*").when({ utilization: 1, preserveRecent: 4 }),
 *
 *     // Per-block: drop error tool results over 500 tokens
 *     Offload.drop("toolResultErrors").when({ threshold: 500 }),
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { Message, TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { ContentBlock } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { LocalAgent } from '../../types/agent.js'
import type { ContextStrategy, ContextState } from '../types.js'
import { truncateToolResultBlock, truncateTextBlock, type TruncateConfig } from './methods/truncate.js'
import {
  flattenMessagesToContent,
  SUMMARIZED_PREFIX,
  summarizeContent,
  toolResultToContentBlocks,
  type SummarizeConfig,
} from './methods/summarize.js'

export const DROPPED_MARKER = '[Dropped]'

/**
 * Target for offload operations. This union is intentionally extensible — new
 * string-literal members can be added freely as new content categories emerge.
 *
 * - `"toolResults"` — all successful tool result blocks
 * - `"toolResultErrors"` — all failed tool result blocks
 * - `"assistantText"` — text blocks in assistant messages
 * - `"userText"` — text blocks in user messages (excluding tool results)
 * - `string[]` — tool results from specific tools, namespaced with `tool::` (e.g. `['tool::bash']`); prefix with `!` to exclude
 * - `"*"` — all content in the context window (tool results + text blocks)
 */
export type OffloadTarget = '*' | 'toolResults' | 'toolResultErrors' | 'assistantText' | 'userText' | string[]

/**
 * Conditions that determine when an offload strategy fires.
 *
 * Granularity is determined by which conditions are set:
 * - `threshold` only → per-block (act on each block above this size, eagerly)
 * - `utilization` without `threshold` → message-level (act on matched set as a whole)
 * - Both → per-block, gated by utilization
 *
 * When multiple strategies target the same content, they don't conflict — strategies
 * run as an ordered pipeline, and once an earlier strategy shrinks a block, it falls
 * below the next strategy's threshold and gets skipped automatically.
 */
export interface OffloadConditions {
  /** Token threshold above which individual blocks are offloaded. */
  threshold?: number

  /** Context utilization ratio (0-1+) above which the strategy fires. */
  utilization?: number

  /** Number of most recent matching messages to leave untouched. */
  preserveRecent?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface OffloadStrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: OffloadConditions): ContextStrategy
}

// --- Shared helpers ---

function finiteOrUndefined(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : undefined
}

/**
 * Checks whether a ToolResultBlock matches the given offload target.
 * Handles status-based targets (toolResults/toolResultErrors) and name-based targets (string[]).
 */
function matchesToolTarget(
  block: ToolResultBlock,
  target: OffloadTarget,
  messages: Message[],
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): boolean {
  if (target === 'toolResults') return block.status === 'success'
  if (target === 'toolResultErrors') return block.status === 'error'

  const toolName = resolveToolName(block, messages)
  if (!toolName) return toolFilter === undefined && excludeFilter === undefined

  if (excludeFilter) return !excludeFilter.has(toolName)
  if (toolFilter) return toolFilter.has(toolName)

  return true
}

// --- Base strategy class ---

/** Shared offload logic: target routing, eager hooks, preserveRecent. */
abstract class BaseOffloadStrategy implements ContextStrategy {
  abstract readonly name: string

  protected readonly _target: OffloadTarget | undefined
  protected readonly _threshold: number | undefined
  protected readonly _utilization: number | undefined
  protected readonly _preserveRecent: number
  protected readonly _toolFilter: Set<string> | undefined
  protected readonly _excludeFilter: Set<string> | undefined
  protected _baseAgent: LocalAgent | undefined

  constructor(target?: OffloadTarget, conditions?: OffloadConditions) {
    this._target = target
    this._threshold = finiteOrUndefined(conditions?.threshold)
    this._utilization = finiteOrUndefined(conditions?.utilization)
    this._preserveRecent = Math.floor(finiteOrUndefined(conditions?.preserveRecent) ?? 0)

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  /** Whether this strategy operates at message-level (batch) vs per-block. */
  protected get _isMessageLevel(): boolean {
    return this._threshold === undefined && this._utilization !== undefined
  }

  init(agent: LocalAgent): void {
    this._baseAgent = agent
    if (this._isMessageLevel) return
    if (this._preserveRecent > 0) return
    agent.addHook(MessageAddedEvent, async (event) => {
      const messages = agent.messages
      await this._transformBlocks(event.message, messages)
    })
  }

  async apply(context: ContextState): Promise<boolean> {
    if (!this._baseAgent) this._baseAgent = context.agent
    if (this._utilization !== undefined && context.utilization < this._utilization) return false

    if (this._isMessageLevel) {
      return this._applyPerMessage(context)
    }

    return this._applyPerBlock(context)
  }

  /** Per-block execution: walk each message, transform individual blocks above threshold. */
  private async _applyPerBlock(context: ContextState): Promise<boolean> {
    const { messages } = context
    const threshold = this._threshold ?? 0
    const eligible =
      this._preserveRecent > 0
        ? getOldestMatches(messages, this._target, this._preserveRecent, this._toolFilter, this._excludeFilter)
        : messages
    let acted = false

    for (const message of eligible) {
      if (await this._transformBlocks(message, messages, threshold)) {
        acted = true
      }
    }

    return acted
  }

  /** Message-level execution: remove oldest 30% of eligible messages with pair safety. */
  protected async _applyPerMessage(context: ContextState): Promise<boolean> {
    const { messages } = context
    if (messages.length <= 1) return false

    const eligible = this._getEligibleMessages(context)
    if (eligible.length === 0) return false

    const targetRemoval = Math.max(1, Math.floor(eligible.length * 0.3))
    const toRemove = eligible.slice(0, targetRemoval)

    const removed = spliceWithPairs(messages, toRemove)
    if (removed > 0) repairAlternation(messages)
    return removed > 0
  }

  /** Process eligible blocks in a message. */
  protected async _transformBlocks(message: Message, messages: Message[], threshold?: number): Promise<boolean> {
    const effectiveThreshold = threshold ?? this._threshold ?? 0
    let acted = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!

      if (block instanceof TextBlock) {
        if (!this._targetIncludesText(message)) continue
      } else if (block instanceof ToolResultBlock) {
        if (
          this._target !== undefined &&
          !matchesToolTarget(block, this._target, messages, this._toolFilter, this._excludeFilter)
        )
          continue
      } else {
        continue
      }

      const role = block instanceof ToolResultBlock ? 'user' : message.role
      const tokens = await this._baseAgent!.model.countTokens([new Message({ role, content: [block] })])
      if (tokens <= effectiveThreshold) continue

      const replacement = await this._replaceBlock(block, tokens, message)
      if (replacement && replacement !== block) {
        ;(message.content as unknown[])[blockIndex] = replacement
        acted = true
      }
    }
    return acted
  }

  private _targetIncludesText(message: Message): boolean {
    if (this._target === '*' || this._target === undefined) return true
    if (this._target === 'assistantText') return message.role === 'assistant'
    if (this._target === 'userText') return message.role === 'user'
    return false
  }

  /** Collect eligible messages for message-level operations, respecting preserveRecent and head-pin. */
  protected _getEligibleMessages(context: ContextState): Message[] {
    const { messages } = context
    if (this._preserveRecent > 0) {
      return getOldestMatches(
        messages,
        this._target,
        this._preserveRecent,
        this._toolFilter,
        this._excludeFilter
      ).filter((message) => messages.indexOf(message) > 0)
    }
    return messages.filter(
      (message, index) =>
        index > 0 && messageMatchesTarget(message, this._target, messages, this._toolFilter, this._excludeFilter)
    )
  }

  /** Transform a block. Return the replacement, or null to skip. */
  protected abstract _replaceBlock(
    block: TextBlock | ToolResultBlock,
    tokens: number,
    message: Message
  ): Promise<ContentBlock | null>
}

// --- Drop strategy ---

class DropStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:drop'

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    _tokens: number,
    message: Message
  ): Promise<ContentBlock | null> {
    if (block instanceof ToolResultBlock) {
      logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from context window`)
      return new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(DROPPED_MARKER)],
      })
    }
    logger.debug(`trackingId=<${message.trackingId}> | dropped text block from context window`)
    return new TextBlock(DROPPED_MARKER)
  }
}

// --- Truncate strategy ---

class TruncateStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:truncate'

  private readonly _truncateConfig: TruncateConfig

  constructor(target?: OffloadTarget, config?: TruncateConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._truncateConfig = config ?? {}

    const previewTokens =
      typeof this._truncateConfig.previewTokens === 'number' && Number.isFinite(this._truncateConfig.previewTokens)
        ? this._truncateConfig.previewTokens
        : 1000
    if (
      conditions?.threshold !== undefined &&
      Number.isFinite(conditions.threshold) &&
      conditions.threshold <= previewTokens
    ) {
      throw new Error(
        `threshold (${conditions.threshold}) must be greater than previewTokens (${previewTokens}) to ensure truncation converges`
      )
    }
  }

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    tokens: number,
    message: Message
  ): Promise<ContentBlock | null> {
    if (block instanceof ToolResultBlock) {
      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | truncated tool result`)
      return truncateToolResultBlock(block, this._truncateConfig)
    }
    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated text block`)
    return truncateTextBlock(block, this._truncateConfig)
  }
}

// --- Summarize strategy ---

class SummarizeStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig
  private _model: Model | undefined

  constructor(target?: OffloadTarget, config?: SummarizeConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._config = config ?? {}
  }

  override init(agent: LocalAgent): void {
    this._baseAgent = agent
    if (this._utilization !== undefined) return
    super.init(agent)
  }

  protected override async _transformBlocks(
    message: Message,
    messages: Message[],
    threshold?: number
  ): Promise<boolean> {
    if (!this._model && this._baseAgent) {
      this._model = this._resolveModel(this._baseAgent)
    }
    return super._transformBlocks(message, messages, threshold)
  }

  override async apply(context: ContextState): Promise<boolean> {
    const model = this._resolveModel(context.agent)
    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }
    this._model = model
    if (context.messages.length === 0) return false
    return super.apply(context)
  }

  protected async _applyPerMessage(context: ContextState): Promise<boolean> {
    if (!this._model) return false

    const { messages } = context
    if (messages.length <= 1) return false

    const eligible = this._getEligibleMessages(context)
    if (eligible.length === 0) return false

    const summarizeCount = Math.max(1, Math.floor(eligible.length * 0.3))
    const toSummarize = eligible.slice(0, summarizeCount)

    // Expand to include paired messages so we don't orphan tool pairs
    const safeSet = new Set<Message>()
    for (const message of toSummarize) {
      const index = messages.indexOf(message)
      if (index === -1) continue
      for (const removable of collectRemovableWithPair(messages, index)) {
        safeSet.add(removable)
      }
    }
    const safe = messages.filter((message) => safeSet.has(message))
    if (safe.length === 0) return false

    const contentBlocks = flattenMessagesToContent(safe)
    const summary = await summarizeContent(contentBlocks, this._model, this._config)
    if (!summary) return false

    const totalTokens = await this._model.countTokens(safe)
    const summaryMessage = new Message({
      role: 'user',
      content: [
        new TextBlock(
          `${SUMMARIZED_PREFIX} ${safe.length} messages, ~${totalTokens.toLocaleString()} tokens]\n\n${summary}`
        ),
      ],
    })

    // Record insertion point before removing (position of the first summarized message)
    const insertIndex = Math.max(1, messages.indexOf(safe[0]!))

    // Remove summarized messages individually (they may not be contiguous)
    for (const message of safe) {
      const index = messages.indexOf(message)
      if (index !== -1) messages.splice(index, 1)
    }

    // Insert summary where the first summarized message was
    const clampedInsert = Math.min(insertIndex, messages.length)
    messages.splice(clampedInsert, 0, summaryMessage)

    repairAlternation(messages)
    logger.debug(`summarized=<${safe.length}>, tokens=<${totalTokens}> | batched summarization complete`)
    return true
  }

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    tokens: number,
    message: Message
  ): Promise<ContentBlock | null> {
    if (!this._model) return null

    if (block instanceof ToolResultBlock) {
      const summary = await summarizeContent(toolResultToContentBlocks(block.content), this._model, this._config)
      if (!summary) return null

      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
      return new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)],
      })
    }

    const summary = await summarizeContent(
      [new TextBlock(`<content>\n${block.text}\n</content>`)],
      this._model,
      this._config
    )
    if (!summary) return null

    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized text block`)
    return new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)
  }

  private _resolveModel(agent: LocalAgent): Model | undefined {
    return this._config.model ?? agent.model
  }
}

// --- preserveRecent helper ---

/**
 * Returns target-matching messages excluding the N most recent matches.
 * First filters to only messages that match the target, then removes the last N from that set.
 */
function getOldestMatches(
  messages: Message[],
  target: OffloadTarget | undefined,
  count: number,
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): Message[] {
  const matching = messages.filter((message) =>
    messageMatchesTarget(message, target, messages, toolFilter, excludeFilter)
  )
  if (count >= matching.length) return []
  return matching.slice(0, -count)
}

/**
 * Checks whether a message matches the given target for preserveRecent counting.
 * A message matches if it contains content that the target would select.
 */
function messageMatchesTarget(
  message: Message,
  target: OffloadTarget | undefined,
  messages: Message[],
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): boolean {
  if (target === undefined || target === '*') return true

  if (target === 'assistantText') return message.role === 'assistant'
  if (target === 'userText') return message.role === 'user'

  // Tool result targets — must be a user message with a matching tool result
  if (message.role !== 'user') return false
  for (const block of message.content) {
    if (block instanceof ToolResultBlock) {
      if (matchesToolTarget(block, target, messages, toolFilter, excludeFilter)) return true
    }
  }
  return false
}

/**
 * Collects a message and its paired partner (if any) for safe removal.
 * If removing a message would orphan a tool-use/tool-result pair, includes the partner
 * so the pair is removed together. Skips messages[0] (head-pin).
 */
function collectRemovableWithPair(messages: Message[], index: number): Message[] {
  const message = messages[index]
  if (!message) return []
  if (index === 0) return []

  const result: Message[] = [message]

  const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
  if (hasToolResult && index > 0) {
    const prev = messages[index - 1]
    if (prev && prev.content.some((block) => block.type === 'toolUseBlock')) {
      if (index - 1 > 0) result.push(prev)
      else return []
    }
  }

  const hasToolUse = message.content.some((block) => block.type === 'toolUseBlock')
  if (hasToolUse && index < messages.length - 1) {
    const next = messages[index + 1]
    if (next && next.content.some((block) => block.type === 'toolResultBlock')) {
      result.push(next)
    }
  }

  return result
}

/**
 * Collects removable messages (with their tool pairs), then splices them from the array.
 * Returns the number of messages removed.
 */
function spliceWithPairs(messages: Message[], toRemove: Message[]): number {
  const toSplice = new Set<Message>()
  for (const message of toRemove) {
    const index = messages.indexOf(message)
    if (index === -1) continue
    for (const removable of collectRemovableWithPair(messages, index)) {
      toSplice.add(removable)
    }
  }

  let removed = 0
  for (const message of toSplice) {
    const index = messages.indexOf(message)
    if (index === -1) continue
    messages.splice(index, 1)
    removed++
  }
  return removed
}

// --- Builder ---

/** Wraps a strategy instance as an OffloadStrategyBuilder with a `.when()` chain. */
function wrapAsBuilder(
  strategy: BaseOffloadStrategy,
  createWithConditions: (conditions: OffloadConditions) => BaseOffloadStrategy
): OffloadStrategyBuilder {
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: OffloadConditions): ContextStrategy {
      return createWithConditions(conditions)
    },
  }
}

/**
 * Offload strategy builder namespace.
 *
 * - `Offload.drop(target)` — drop matching content from context window entirely
 * - `Offload.truncate(target, config)` — replace with a preview
 * - `Offload.summarize(target, config)` — replace with LLM-generated summary
 */
interface OffloadNamespace {
  /** Drop matching content from context window entirely. */
  drop(target: OffloadTarget): OffloadStrategyBuilder

  /** Replace oversized content with a preview. */
  truncate(target: OffloadTarget, config?: TruncateConfig): OffloadStrategyBuilder

  /** Replace oversized content with an LLM-generated summary. */
  summarize(target: OffloadTarget, config?: SummarizeConfig): OffloadStrategyBuilder
}

/**
 * Builder for offload strategies — reduces content in the context window.
 *
 * @example
 * ```typescript
 * // Per-block: truncate each result over 2500 tokens, eagerly
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * // Per-block: truncate specific tools
 * Offload.truncate(["tool::bash", "tool::read_file"]).when({ threshold: 2000 })
 * // Message-level: summarize oldest messages on overflow
 * Offload.summarize("*").when({ utilization: 1, preserveRecent: 4 })
 * // Per-block: drop errors over 500 tokens
 * Offload.drop("toolResultErrors").when({ threshold: 500 })
 * ```
 */
export const Offload: OffloadNamespace = {
  drop(target: OffloadTarget): OffloadStrategyBuilder {
    return wrapAsBuilder(new DropStrategy(target), (c) => new DropStrategy(target, c))
  },

  truncate(target: OffloadTarget, config?: TruncateConfig): OffloadStrategyBuilder {
    return wrapAsBuilder(new TruncateStrategy(target, config), (c) => new TruncateStrategy(target, config, c))
  },

  summarize(target: OffloadTarget, config?: SummarizeConfig): OffloadStrategyBuilder {
    return wrapAsBuilder(new SummarizeStrategy(target, config), (c) => new SummarizeStrategy(target, config, c))
  },
}

// --- Role alternation repair ---

/**
 * Merges consecutive same-role messages to restore the user/assistant alternation
 * that Anthropic/Bedrock APIs require. Called after message-level operations that
 * may leave gaps.
 */
function repairAlternation(messages: Message[]): void {
  let writeIndex = 0
  for (let readIndex = 0; readIndex < messages.length; readIndex++) {
    const current = messages[readIndex]!
    if (writeIndex > 0 && messages[writeIndex - 1]!.role === current.role) {
      const prev = messages[writeIndex - 1]!
      ;(prev.content as ContentBlock[]).push(...current.content)
    } else {
      messages[writeIndex] = current
      writeIndex++
    }
  }
  messages.length = writeIndex
}

// --- Helpers ---

/**
 * Resolves the tool name for a ToolResultBlock by finding the corresponding ToolUseBlock
 * in the preceding assistant message (where ToolUseBlocks live).
 */
function resolveToolName(block: ToolResultBlock, messages: Message[]): string | undefined {
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const content of message.content) {
      if (
        'toolUseId' in content &&
        'name' in content &&
        (content as { toolUseId: string }).toolUseId === block.toolUseId
      ) {
        return (content as { name: string }).name
      }
    }
  }
  return undefined
}

/**
 * Parses a string[] target into include/exclude filter sets.
 * Entries must be prefixed with `tool::` (e.g. `'tool::bash'`).
 * An additional `!` prefix excludes (e.g. `'!tool::bash'`).
 */
function resolveToolFilter(target: OffloadTarget | undefined): { include?: Set<string>; exclude?: Set<string> } {
  if (target === undefined || target === '*') return {}
  if (typeof target === 'string') return {}
  if (!Array.isArray(target)) return {}

  const TOOL_PREFIX = 'tool::'
  const includes: string[] = []
  const excludes: string[] = []

  for (const entry of target) {
    if (entry.startsWith('!')) {
      const name = entry.slice(1)
      excludes.push(name.startsWith(TOOL_PREFIX) ? name.slice(TOOL_PREFIX.length) : name)
    } else {
      includes.push(entry.startsWith(TOOL_PREFIX) ? entry.slice(TOOL_PREFIX.length) : entry)
    }
  }

  if (excludes.length > 0 && includes.length > 0) {
    logger.warn('tool filter contains both includes and excludes, excludes will be ignored')
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
