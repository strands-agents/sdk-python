import { Message, TextBlock } from '../../../types/messages.js'
import type { Model } from '../../../models/model.js'
import { isProtected } from '../protection.js'
import { logger } from '../../../logging/logger.js'

const SUMMARIZATION_PROMPT = `You are a conversation summarizer. Provide a concise summary of the conversation history.

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

export type SummarizeOptions = {
  /** Ratio of messages to summarize (0.1–0.8). Defaults to 0.3. */
  summaryRatio?: number
  /** Minimum recent messages to preserve. Defaults to 10. */
  preserveRecentMessages?: number
  /** Positive: protect first N messages. Negative: protect last N messages. */
  protectFirst?: number
}

/**
 * Summarize the oldest messages and replace them with a model-generated summary.
 *
 * @param messages - The messages array to mutate in place
 * @param model - The model to use for generating the summary
 * @param options - Summarization options
 * @returns `true` if messages were summarized, `false` if not enough to summarize
 */
export async function summarize(messages: Message[], model: Model, options?: SummarizeOptions): Promise<boolean> {
  const summaryRatio = Math.max(0.1, Math.min(0.8, options?.summaryRatio ?? 0.3))
  const preserveRecent = options?.preserveRecentMessages ?? 10
  const protectedRange = options?.protectFirst

  let count = Math.max(1, Math.floor(messages.length * summaryRatio))
  count = Math.min(count, messages.length - preserveRecent)

  if (count <= 0) {
    logger.warn(
      `preserve_recent=<${preserveRecent}>, messages=<${messages.length}> | insufficient messages for summarization`
    )
    return false
  }

  count = adjustSplitForToolPairs(messages, count)

  // Identify protected messages that must survive verbatim
  const protectedToPreserve: Message[] = []
  let hasUnprotected = false
  for (let i = 0; i < count; i++) {
    if (isProtected(messages, i, protectedRange)) {
      protectedToPreserve.push(messages[i]!)
    } else {
      hasUnprotected = true
    }
  }

  if (!hasUnprotected) {
    logger.warn(`messages=<${messages.length}> | all messages in summarize range are protected, unable to reduce`)
    return false
  }

  // Summarize ALL messages in range (including protected) for full context
  const summary = await generateSummary(messages.slice(0, count), model)

  // Replace range with protected messages (verbatim) + summary
  messages.splice(0, count, ...protectedToPreserve, summary)
  return true
}

async function generateSummary(messagesToSummarize: Message[], model: Model): Promise<Message> {
  const input = [
    ...messagesToSummarize,
    new Message({ role: 'user', content: [new TextBlock('Please summarize this conversation.')] }),
  ]

  const stream = model.streamAggregated(input, { systemPrompt: SUMMARIZATION_PROMPT })

  let result: Awaited<ReturnType<typeof stream.next>> | undefined
  for (;;) {
    result = await stream.next()
    if (result.done) break
  }

  if (!result?.done || !result.value) {
    throw new Error('Failed to generate summary: no response from model')
  }

  const summaryText = result.value.message.content
    .filter((block) => block.type === 'textBlock')
    .map((block) => (block as TextBlock).text)
    .join('\n')

  return new Message({
    role: 'user',
    content: [new TextBlock(`<summary>\n${summaryText}\n</summary>`)],
  })
}

/**
 * Adjust split point forward to avoid breaking tool use/result pairs.
 */
function adjustSplitForToolPairs(messages: Message[], splitPoint: number): number {
  if (splitPoint >= messages.length) return splitPoint

  let idx = splitPoint
  while (idx < messages.length) {
    const msg = messages[idx]!

    if (msg.content.some((b) => b.type === 'toolResultBlock')) {
      idx++
      continue
    }

    const hasToolUse = msg.content.some((b) => b.type === 'toolUseBlock')
    if (hasToolUse) {
      const next = messages[idx + 1]
      if (!next?.content.some((b) => b.type === 'toolResultBlock')) {
        idx++
        continue
      }
    }

    break
  }

  return idx >= messages.length ? splitPoint : idx
}
