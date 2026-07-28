/**
 * Summarize reduction method.
 *
 * Compresses a slice of messages into a single summary via an LLM call.
 * Used by both Offload (to compress oldest messages in L0) and Inject
 * (to inject a summarized view of stored content).
 *
 * @internal
 */

import type { Model } from '../../../models/model.js'
import type { Message } from '../../../types/messages.js'
import {
  adjustSplitPointForToolPairs,
  generateSummary,
} from '../../../conversation-manager/compression/context-compression.js'
import { logger } from '../../../logging/logger.js'

/**
 * Configuration for the summarize method.
 */
export interface SummarizeConfig {
  /** Ratio of messages to summarize (0.1 - 0.8). Defaults to 0.3. */
  ratio?: number

  /** Number of recent messages to always preserve. Defaults to 10. */
  preserveRecent?: number

  /** Model to use for summarization. When omitted, uses the agent's model. */
  model?: Model

  /** Custom system prompt for the summarization model. */
  systemPrompt?: string
}

/**
 * Summarizes the oldest portion of a message array in place.
 *
 * @returns true if messages were summarized, false if conditions weren't met.
 */
export async function summarizeMessages(
  messages: Message[],
  model: Model,
  config?: SummarizeConfig
): Promise<boolean> {
  const ratio = Math.max(0.1, Math.min(0.8, config?.ratio ?? 0.3))
  const preserveRecent = config?.preserveRecent ?? 10

  let messagesToSummarize = Math.max(1, Math.floor(messages.length * ratio))
  messagesToSummarize = Math.min(messagesToSummarize, messages.length - preserveRecent)

  if (messagesToSummarize <= 0) {
    logger.debug(
      `preserveRecent=<${preserveRecent}>, messages=<${messages.length}> | insufficient messages for summarization`
    )
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
    const summaryMessage = await generateSummary(toSummarize, model, config?.systemPrompt)
    messages.splice(0, messagesToSummarize, summaryMessage)

    logger.debug(
      `summarized=<${messagesToSummarize}>, remaining=<${messages.length}> | summarized oldest messages`
    )
    return true
  } catch (error) {
    logger.warn(`error=<${error}> | summarization failed`)
    return false
  }
}
