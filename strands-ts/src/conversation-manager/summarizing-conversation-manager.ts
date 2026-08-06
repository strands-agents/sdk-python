/**
 * Summarization-based conversation history management.
 *
 * This module provides a conversation manager that summarizes older messages
 * when the context window overflows, preserving important information rather
 * than simply discarding it.
 */

import type { LocalAgent } from '../types/agent.js'
import {
  ConversationManager,
  type ProactiveCompressionConfig,
  type ConversationManagerReduceOptions,
} from './conversation-manager.js'
import { applyPinFirst, partitionPinned } from './compression/pin-message.js'
import {
  adjustSplitPointForToolPairs,
  generateSummary,
  generateSummaryCacheAligned,
  DEFAULT_SUMMARIZATION_PROMPT,
} from './compression/context-compression.js'
import type { Message } from '../types/messages.js'
import { logger } from '../logging/logger.js'
import { normalizeError } from '../errors.js'
import type { Model } from '../models/model.js'

/**
 * Configuration for the summarization conversation manager.
 */
export type SummarizingConversationManagerConfig = {
  /**
   * Model to use for generating summaries. When provided, overrides the model
   * attached to the agent. Useful when you want to use a different model than
   * the one attached to the agent.
   */
  model?: Model

  /**
   * Ratio of messages to summarize when context overflow occurs.
   * Value is clamped to [0.1, 0.8]. Defaults to 0.3 (summarize 30% of oldest messages).
   */
  summaryRatio?: number

  /**
   * Minimum number of recent messages to always keep.
   * Defaults to 10.
   */
  preserveRecentMessages?: number

  /**
   * Custom system prompt for summarization. If not provided, uses a default
   * prompt that produces structured bullet-point summaries.
   */
  summarizationSystemPrompt?: string

  /**
   * Enable proactive context compression before the model call.
   *
   * - `true`: compress when 70% of the context window is used (default threshold).
   * - `{ compressionThreshold: number }`: compress at the specified ratio (0, 1].
   * - `false` or omitted: disabled, only reactive overflow recovery is used.
   */
  proactiveCompression?: boolean | ProactiveCompressionConfig

  /**
   * Number of messages at the start of the conversation to permanently pin.
   * Pinned messages are protected from summarization and compacted to the front.
   */
  pinFirst?: number

  /**
   * Align the summarization request with the live conversation's request so the
   * provider prompt cache is reused.
   *
   * When `true`, proactive summarization reuses the agent's live system prompt
   * and tool specs and the full message history, delivering the summarization
   * instruction at the tail of the request. This makes the request prefix
   * byte-identical to the live conversation's request, so the provider prompt
   * cache is hit instead of paying to re-read the history.
   *
   * Only enable this when the model has prompt caching configured (e.g.
   * `cacheConfig` on `BedrockModel`): the aligned request re-sends the entire
   * history, so without a cache hit it is strictly more expensive than the
   * slice-based path.
   *
   * Note a semantic difference from the slice-based path: the model sees the
   * entire history — including the recent messages that are preserved verbatim —
   * rather than only the oldest slice being summarized. The default instruction
   * tells the model those recent messages are preserved and should be excluded
   * from the summary.
   *
   * Only takes effect on proactive compression, and is ignored when a `model`
   * override is configured (a different model cannot reuse the live model's
   * cache). Reactive overflow recovery always uses slice-based summarization:
   * the cache-aligned request is a superset of the live request, so it only
   * fits under the proactive threshold — on overflow it would overflow again.
   * Defaults to `false` (existing slice-based behavior).
   */
  cacheAligned?: boolean
}

/**
 * Implements a summarization strategy for managing conversation history.
 *
 * When a {@link ContextWindowOverflowError} occurs, this manager summarizes
 * the oldest messages using a model call and replaces them with a single
 * summary message, preserving context that would otherwise be lost.
 */
export class SummarizingConversationManager extends ConversationManager {
  readonly name = 'strands:summarizing-conversation-manager'

  private readonly _model: Model | undefined
  private readonly _summaryRatio: number
  private readonly _preserveRecentMessages: number
  private readonly _summarizationSystemPrompt: string
  /**
   * The raw user-supplied `summarizationSystemPrompt`, without the default applied.
   * The cache-aligned path must distinguish "user configured a prompt" (use it as the
   * trailing instruction) from "default" (use {@link DEFAULT_SUMMARIZATION_INSTRUCTION},
   * which is purpose-built for user-turn delivery — the default system prompt is not).
   */
  private readonly _customSummarizationPrompt: string | undefined
  private readonly _pinFirst: number | undefined
  private readonly _cacheAligned: boolean
  private _pinFirstApplied = false

  constructor(config?: SummarizingConversationManagerConfig) {
    super(config)
    this._model = config?.model
    // clamped [0.1, 0.8]
    this._summaryRatio = Math.max(0.1, Math.min(0.8, config?.summaryRatio ?? 0.3))
    this._preserveRecentMessages = config?.preserveRecentMessages ?? 10
    this._customSummarizationPrompt = config?.summarizationSystemPrompt
    this._summarizationSystemPrompt = config?.summarizationSystemPrompt ?? DEFAULT_SUMMARIZATION_PROMPT
    this._pinFirst = config?.pinFirst != null ? Math.max(0, config.pinFirst) : undefined
    this._cacheAligned = config?.cacheAligned ?? false

    if (this._cacheAligned && config?.model) {
      logger.warn(
        'cache_aligned=<true> | model override defeats cache alignment | summaries will miss the prompt cache'
      )
    }
    if (this._cacheAligned && this._compressionThreshold === undefined) {
      logger.warn(
        'cache_aligned=<true> | cacheAligned only takes effect on proactive compression | reactive overflow uses slice-based summarization'
      )
    }
  }

  /**
   * Reduce the conversation history by summarizing older messages.
   *
   * When `error` is set (reactive overflow recovery), summarization failure is rethrown
   * with the original error as cause — the agent loop must not proceed with an overflow.
   *
   * When `error` is undefined (proactive compression), summarization failure is logged
   * and returns `false` — the model call proceeds regardless.
   *
   * @param options - The reduction options
   * @returns `true` if the history was reduced, `false` otherwise
   */
  async reduce({ agent, model, error }: ConversationManagerReduceOptions): Promise<boolean> {
    try {
      // `error === undefined` means proactive compression; a set error means reactive overflow recovery.
      return await this._summarizeOldest(agent, this._model ?? model, error === undefined)
    } catch (summarizationError) {
      if (error) {
        // Reactive: rethrow so the ContextWindowOverflowError propagates
        logger.error(`error=<${summarizationError}> | summarization failed`)
        const wrapped = normalizeError(summarizationError)
        wrapped.cause = error
        throw wrapped
      }
      // Proactive: best-effort, swallow errors so the model call can still proceed.
      logger.warn(`error=<${summarizationError}> | proactive summarization failed, continuing`)
      return false
    }
  }

  /**
   * Summarize the oldest messages and replace them with a summary.
   *
   * @param agent - The agent instance
   * @param model - The model to use for summarization
   * @param proactive - `true` for proactive compression, `false` for reactive overflow recovery
   * @returns `true` if the history was reduced, `false` otherwise
   */
  private async _summarizeOldest(agent: LocalAgent, model: Model, proactive: boolean): Promise<boolean> {
    const messages = agent.messages

    // Calculate how many messages to summarize
    let messagesToSummarizeCount = Math.max(1, Math.floor(messages.length * this._summaryRatio))

    // Don't touch recent messages
    messagesToSummarizeCount = Math.min(messagesToSummarizeCount, messages.length - this._preserveRecentMessages)

    if (messagesToSummarizeCount <= 0) {
      logger.warn(
        `preserve_recent=<${this._preserveRecentMessages}>, messages=<${messages.length}> | insufficient messages for summarization`
      )
      return false
    }

    // Adjust split point to avoid breaking tool use/result pairs
    messagesToSummarizeCount = adjustSplitPointForToolPairs(messages, messagesToSummarizeCount)

    // Pin first N messages permanently (only on first reduction)
    if (this._pinFirst && !this._pinFirstApplied) {
      applyPinFirst(messages, this._pinFirst)
      this._pinFirstApplied = true
    }

    // Partition [0, messagesToSummarizeCount) into pinned (preserve) and non-pinned (summarize)
    const [protectedToPreserve, toSummarize] = partitionPinned(messages, 0, messagesToSummarizeCount)

    if (toSummarize.length === 0) {
      logger.warn(`messages=<${messages.length}> | all messages in summarize range are protected, unable to reduce`)
      return false
    }

    // Generate the summary. The cache-aligned path sends the full history with the agent's live
    // system prompt + tool specs so the request prefix matches the live turn and hits the prompt
    // cache. It only fits under the proactive threshold: the cache-aligned request is a superset of
    // the live request, so on reactive overflow it would overflow again — hence the slice-based
    // fallback there (and when cacheAligned is off, or the tail can't carry the instruction).
    // A configured model override also disables the aligned path: sending the full live history to
    // a different model guarantees a cache miss, making it strictly worse than slice-based.
    let summaryMessage: Message | undefined
    if (this._cacheAligned && proactive && this._model === undefined) {
      if (this._isValidAppendTail(messages)) {
        try {
          summaryMessage = await generateSummaryCacheAligned(messages, model, {
            systemPrompt: agent.systemPrompt,
            toolSpecs: agent.toolRegistry.list().map((tool) => tool.toolSpec),
            // Only a user-supplied prompt overrides the instruction; otherwise
            // DEFAULT_SUMMARIZATION_INSTRUCTION applies inside generateSummaryCacheAligned.
            instruction: this._customSummarizationPrompt,
          })
        } catch (alignedError) {
          // Without this fallback no compaction would happen, and a persistently failing aligned
          // request would repeat on every subsequent turn.
          logger.warn(
            `error=<${alignedError}> | cache-aligned summarization failed | falling back to slice-based summarization`
          )
        }
      } else {
        logger.debug(
          'cache_aligned=<true> | message tail is not a valid append point | falling back to slice-based summarization'
        )
      }
    }
    summaryMessage ??= await generateSummary(toSummarize, model, this._summarizationSystemPrompt)

    // Replace summarized range with protected messages + summary. Recent messages stay verbatim;
    // only the oldest range is replaced, regardless of how the summary was generated.
    messages.splice(0, messagesToSummarizeCount, ...protectedToPreserve, summaryMessage)

    return true
  }

  /**
   * Returns `true` if the summarization instruction can be delivered at the tail —
   * merged into a trailing user message, or appended as a new user turn after an
   * assistant text message — i.e. the last message is not an assistant toolUse
   * still awaiting its toolResult. Empty histories are valid.
   */
  private _isValidAppendTail(messages: Message[]): boolean {
    const last = messages[messages.length - 1]
    if (!last) {
      return true
    }
    const hasToolUse = last.content.some((block) => block.type === 'toolUseBlock')
    const hasToolResult = last.content.some((block) => block.type === 'toolResultBlock')
    return !(hasToolUse && !hasToolResult)
  }
}
