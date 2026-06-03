/**
 * Summarization-based conversation history management.
 *
 * This module provides a conversation manager that summarizes older messages
 * when the context window overflows, preserving important information rather
 * than simply discarding it.
 */

import { Message, TextBlock } from '../types/messages.js'
import type { LocalAgent } from '../types/agent.js'
import {
  ConversationManager,
  type ProactiveCompressionConfig,
  type ConversationManagerReduceOptions,
} from './conversation-manager.js'
import { logger } from '../logging/logger.js'
import { normalizeError } from '../errors.js'
import type { Model } from '../models/model.js'

const DEFAULT_SUMMARIZATION_PROMPT = `You are a conversation summarizer. Provide a concise summary of the conversation \
history.

Format Requirements:
- You MUST create a structured and concise summary in bullet-point format.
- You MUST NOT respond conversationally.
- You MUST NOT address the user directly.
- You MUST NOT comment on tool availability.

Assumptions:
- You MUST NOT assume tool executions failed unless otherwise stated.

Task:
Your task is to create a structured summary document:
- It MUST contain bullet points with key topics and questions covered
- It MUST contain bullet points for all significant tools executed and their results
- It MUST contain bullet points for any code or technical information shared
- It MUST contain a section of key insights gained
- It MUST format the summary in the third person

Example format:

## Conversation Summary
* Topic 1: Key information
* Topic 2: Key information

## Tools Executed
* Tool X: Result Y`

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

  constructor(config?: SummarizingConversationManagerConfig) {
    super(config)
    this._model = config?.model
    // clamped [0.1, 0.8]
    this._summaryRatio = Math.max(0.1, Math.min(0.8, config?.summaryRatio ?? 0.3))
    this._preserveRecentMessages = config?.preserveRecentMessages ?? 10
    this._summarizationSystemPrompt = config?.summarizationSystemPrompt ?? DEFAULT_SUMMARIZATION_PROMPT
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
      return await this._summarizeOldest(agent, this._model ?? model)
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
   * @returns `true` if the history was reduced, `false` otherwise
   */
  private async _summarizeOldest(agent: LocalAgent, model: Model): Promise<boolean> {
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
    messagesToSummarizeCount = this._adjustSplitPointForToolPairs(messages, messagesToSummarizeCount)

    const messagesToSummarize = messages.slice(0, messagesToSummarizeCount)

    // Generate summary via model call
    const summaryMessage = await this._generateSummary(messagesToSummarize, model)

    // Replace summarized messages with the summary
    messages.splice(0, messagesToSummarizeCount, summaryMessage)

    return true
  }

  /**
   * Generate a summary of the provided messages by calling the model directly.
   *
   * @param messagesToSummarize - The messages to summarize
   * @returns A user-role message containing the summary
   */
  private async _generateSummary(messagesToSummarize: Message[], model: Model): Promise<Message> {
    const summarizationMessages = [
      ...messagesToSummarize,
      new Message({
        role: 'user',
        content: [new TextBlock('Please summarize this conversation.')],
      }),
    ]

    const stream = model.streamAggregated(summarizationMessages, {
      systemPrompt: this._summarizationSystemPrompt,
    })

    // Manual .next() loop is required: streamAggregated returns its final result
    // as the generator return value (done:true), which for-await-of discards.
    let result: Awaited<ReturnType<typeof stream.next>> | undefined
    for (;;) {
      result = await stream.next()
      if (result.done) break
    }

    if (!result?.done || !result.value) {
      throw new Error('Failed to generate summary: no response from model')
    }

    // Return the summary as a user-role message so it's valid as conversation history
    return new Message({
      role: 'user',
      content: result.value.message.content,
    })
  }

  /**
   * Adjust the split point to avoid breaking tool use/result pairs.
   *
   * Walks the split point forward until the message at that position is neither
   * an orphaned toolResult nor a toolUse without an immediately following toolResult.
   *
   * @param messages - The full message array
   * @param splitPoint - The initially calculated split point
   * @returns The adjusted split point
   * @throws If no valid split point can be found
   */
  private _adjustSplitPointForToolPairs(messages: Message[], splitPoint: number): number {
    if (splitPoint >= messages.length) {
      return splitPoint
    }

    while (splitPoint < messages.length) {
      const message = messages[splitPoint]!

      // Can't leave an orphaned toolResult at the start
      const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
      if (hasToolResult) {
        splitPoint++
        continue
      }

      // A toolUse is only valid at the boundary if the next message is its toolResult
      const hasToolUse = message.content.some((block) => block.type === 'toolUseBlock')
      if (hasToolUse) {
        const nextMessage = messages[splitPoint + 1]
        const nextHasToolResult = nextMessage?.content.some((block) => block.type === 'toolResultBlock')
        if (!nextHasToolResult) {
          splitPoint++
          continue
        }
      }

      break
    }

    // If we walked past all messages, no valid split point exists
    if (splitPoint >= messages.length) {
      throw new Error('Unable to find valid split point for summarization')
    }

    return splitPoint
  }
}
