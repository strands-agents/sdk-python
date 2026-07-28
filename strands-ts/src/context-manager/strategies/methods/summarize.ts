/**
 * Summarize reduction method.
 *
 * Compresses text via an LLM call. The method is target-agnostic — it operates
 * on any text (a tool result, a range of messages, raw storage content).
 * Strategies handle selection and placement.
 *
 * @internal
 */

import type { Model } from '../../../models/model.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { logger } from '../../../logging/logger.js'

// Subject to change as we benchmark summarization quality.
const DEFAULT_SYSTEM_PROMPT = [
  'You are a summarization assistant. Produce a concise summary that preserves:',
  '- Key decisions and conclusions',
  '- Important data, values, and identifiers',
  '- Action items and next steps',
  '- Context needed to continue the work',
  '',
  'Be concise. Omit pleasantries, repetition, and obvious context.',
].join('\n')

/**
 * Configuration for the summarize method.
 */
export interface SummarizeConfig {
  /** Model to use for summarization. When omitted, uses the agent's model. */
  model?: Model

  /** Custom system prompt for the summarization model. */
  systemPrompt?: string
}

/**
 * Summarizes arbitrary text into a shorter form via an LLM call.
 *
 * @returns The summarized text, or null if summarization failed.
 */
export async function summarizeText(
  text: string,
  model: Model,
  config?: SummarizeConfig
): Promise<string | null> {
  const messages = [
    new Message({
      role: 'user',
      content: [new TextBlock(`Please summarize the following content:\n\n${text}`)],
    }),
  ]

  try {
    const stream = model.streamAggregated(messages, {
      systemPrompt: config?.systemPrompt ?? DEFAULT_SYSTEM_PROMPT,
    })

    let result: Awaited<ReturnType<typeof stream.next>> | undefined
    for (;;) {
      result = await stream.next()
      if (result.done) break
    }

    if (!result?.done || !result.value) {
      logger.warn('summarization produced no response')
      return null
    }

    const outputBlocks = result.value.message.content
    const parts: string[] = []
    for (const block of outputBlocks) {
      if (block instanceof TextBlock) {
        parts.push(block.text)
      }
    }

    return parts.join('\n') || null
  } catch (error) {
    logger.warn(`error=<${error}> | summarization failed`)
    return null
  }
}
