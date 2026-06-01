import type { Message } from '../types/messages.js'
import type { Model } from '../models/model.js'
import { logger } from '../logging/logger.js'

/**
 * Estimate input tokens for a conversation.
 *
 * Uses an incremental strategy: if the last assistant message has usage metadata,
 * uses (inputTokens + outputTokens) as a baseline and only counts new messages
 * added after it. Falls back to full model estimation otherwise.
 *
 * @param messages - The conversation messages
 * @param model - The model to use for token counting
 * @returns Estimated token count, or undefined if estimation fails
 */
export async function estimateInputTokens(messages: Message[], model: Model): Promise<number | undefined> {
  try {
    for (let i = messages.length - 1; i >= 0; i--) {
      const usage = messages[i]!.metadata?.usage
      if (messages[i]!.role === 'assistant' && usage) {
        const baseline = usage.inputTokens + usage.outputTokens
        const newMessages = messages.slice(i + 1)
        return newMessages.length === 0 ? baseline : baseline + (await model.countTokens(newMessages))
      }
    }
    return await model.countTokens(messages)
  } catch (e) {
    logger.debug(`error=<${e}> | token estimation failed`)
    return undefined
  }
}
