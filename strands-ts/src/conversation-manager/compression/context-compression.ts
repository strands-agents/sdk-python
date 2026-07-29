import { Message, TextBlock } from '../../types/messages.js'
import type { SystemPrompt } from '../../types/messages.js'
import type { Model, StreamOptions } from '../../models/model.js'
import type { ToolSpec } from '../../tools/types.js'

/**
 * Requirements shared by {@link DEFAULT_SUMMARIZATION_PROMPT} and
 * {@link DEFAULT_SUMMARIZATION_INSTRUCTION} — the two texts differ only in
 * their opening line (persona vs. imperative) and the trailing example block.
 */
const SUMMARIZATION_REQUIREMENTS = `Format Requirements:
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
- It MUST format the summary in the third person`

export const DEFAULT_SUMMARIZATION_PROMPT = `You are a conversation summarizer. Provide a concise summary of the conversation \
history.

${SUMMARIZATION_REQUIREMENTS}

Example format:

## Conversation Summary
* Topic 1: Key information
* Topic 2: Key information

## Tools Executed
* Tool X: Result Y`

/**
 * Default instruction delivered at the tail of a cache-aligned summary request
 * (merged into the final user message, or appended as a new user turn).
 *
 * Unlike {@link DEFAULT_SUMMARIZATION_PROMPT} (which is delivered as a system
 * prompt), this text is delivered as normal user content so the request
 * prefix — system prompt, tool specs, and message history — stays byte-identical
 * to the live conversation's request and the provider prompt cache is reused.
 * It shares {@link SUMMARIZATION_REQUIREMENTS} with the system-prompt variant
 * rather than relying on a summarization-specific system prompt.
 */
export const DEFAULT_SUMMARIZATION_INSTRUCTION = `Summarize the conversation so far. The most recent messages will be \
preserved verbatim alongside the summary, so exclude them from the summary and focus on the earlier conversation.

${SUMMARIZATION_REQUIREMENTS}`

/**
 * Adjust a split point forward to avoid breaking tool use/result pairs.
 *
 * Walks the split point forward until the message at that position is neither
 * an orphaned toolResult nor a toolUse without an immediately following toolResult.
 *
 * @throws If no valid split point can be found (walked past all messages)
 */
export function adjustSplitPointForToolPairs(messages: Message[], splitPoint: number): number {
  if (splitPoint >= messages.length) {
    return splitPoint
  }

  while (splitPoint < messages.length) {
    const message = messages[splitPoint]!

    const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
    if (hasToolResult) {
      splitPoint++
      continue
    }

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

  if (splitPoint >= messages.length) {
    throw new Error('Unable to find valid split point for summarization')
  }

  return splitPoint
}

/**
 * Find a valid trim point for truncation starting at `startIndex`.
 *
 * A valid trim point must:
 * 1. Be a user message (required by some models)
 * 2. Not be an orphaned toolResult
 * 3. Not be a toolUse unless its toolResult immediately follows
 *
 * @returns The valid trim index, or `messages.length` if none found
 */
export function findValidTrimPoint(messages: Message[], startIndex: number): number {
  let trimIndex = startIndex

  while (trimIndex < messages.length) {
    const message = messages[trimIndex]
    if (!message) break

    if (message.role !== 'user') {
      trimIndex++
      continue
    }

    const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
    if (hasToolResult) {
      trimIndex++
      continue
    }

    const hasToolUse = message.content.some((block) => block.type === 'toolUseBlock')
    if (hasToolUse) {
      const nextMessage = messages[trimIndex + 1]
      const nextHasToolResult = nextMessage && nextMessage.content.some((block) => block.type === 'toolResultBlock')
      if (!nextHasToolResult) {
        trimIndex++
        continue
      }
    }

    break
  }

  return trimIndex
}

/**
 * Drain an aggregated model stream and wrap the final response in a
 * user-role summary message.
 *
 * @throws If the model fails to produce a response
 */
async function drainSummaryResponse(stream: ReturnType<Model['streamAggregated']>): Promise<Message> {
  let result: Awaited<ReturnType<typeof stream.next>> | undefined
  for (;;) {
    result = await stream.next()
    if (result.done) break
  }

  if (!result?.done || !result.value) {
    throw new Error('Failed to generate summary: no response from model')
  }

  return new Message({
    role: 'user',
    content: result.value.message.content,
  })
}

/**
 * Generate a summary of the provided messages by calling the model.
 *
 * @returns A user-role message containing the model-generated summary
 * @throws If the model fails to produce a response
 */
export async function generateSummary(
  messagesToSummarize: Message[],
  model: Model,
  systemPrompt?: string
): Promise<Message> {
  const summarizationMessages = [
    ...messagesToSummarize,
    new Message({
      role: 'user',
      content: [new TextBlock('Please summarize this conversation.')],
    }),
  ]

  return await drainSummaryResponse(
    model.streamAggregated(summarizationMessages, {
      systemPrompt: systemPrompt ?? DEFAULT_SUMMARIZATION_PROMPT,
    })
  )
}

/**
 * Options for {@link generateSummaryCacheAligned}.
 */
export interface CacheAlignedSummaryOptions {
  /**
   * The agent's live system prompt, reused verbatim so the request prefix
   * matches the live conversation's request.
   */
  systemPrompt?: SystemPrompt | undefined

  /**
   * The agent's live tool specs, reused verbatim so the request prefix
   * matches the live conversation's request.
   */
  toolSpecs?: ToolSpec[] | undefined

  /**
   * Summarization instruction delivered at the tail of the request.
   * Defaults to {@link DEFAULT_SUMMARIZATION_INSTRUCTION}.
   */
  instruction?: string | undefined
}

/**
 * Generate a summary using a cache-aligned request.
 *
 * The request reuses the agent's live system prompt and tool specs and the full
 * message history, delivering the summarization instruction at the tail. This
 * keeps the request prefix byte-identical to the live conversation's request so
 * the provider prompt cache is reused, rather than sending a fresh
 * summarization-only request that would miss the cache.
 *
 * When the history ends with a user message (always the case at the proactive
 * compression trigger — the upcoming turn's input is already appended), the
 * instruction is merged into a clone of that final user message as an
 * additional text block. Appending it as a new turn would produce consecutive
 * user roles, which providers like Bedrock reject for Anthropic models. The
 * final message lies beyond the previous turn's cache point, so the cached
 * prefix `messages[0..n-2]` is untouched either way. When the history ends
 * with an assistant message, the instruction is appended as a new user turn.
 *
 * @returns A user-role message containing the model-generated summary
 * @throws If the model fails to produce a response, or responds with tool use
 *   instead of a text summary
 */
export async function generateSummaryCacheAligned(
  allMessages: Message[],
  model: Model,
  options: CacheAlignedSummaryOptions
): Promise<Message> {
  const instructionBlock = new TextBlock(options.instruction ?? DEFAULT_SUMMARIZATION_INSTRUCTION)

  const last = allMessages[allMessages.length - 1]
  let request: Message[]
  if (last && last.role === 'user') {
    // Clone rather than mutate: the live history must not carry the instruction.
    const merged = new Message({
      role: 'user',
      content: [...last.content, instructionBlock],
      ...(last.metadata !== undefined && { metadata: last.metadata }),
    })
    request = [...allMessages.slice(0, -1), merged]
  } else {
    request = [...allMessages, new Message({ role: 'user', content: [instructionBlock] })]
  }

  // Only set keys that are defined so the request prefix (system prompt, tool specs) matches the
  // live turn exactly under exactOptionalPropertyTypes.
  const streamOptions: StreamOptions = {}
  if (options.systemPrompt !== undefined) {
    streamOptions.systemPrompt = options.systemPrompt
  }
  if (options.toolSpecs !== undefined) {
    streamOptions.toolSpecs = options.toolSpecs
  }

  const summary = await drainSummaryResponse(model.streamAggregated(request, streamOptions))

  // The aligned request carries the live tool specs, so the model may answer with a tool call
  // instead of a summary. Splicing that into history would leave a dangling toolUse without a
  // toolResult — reject it so the caller can fall back to slice-based summarization.
  if (summary.content.some((block) => block.type === 'toolUseBlock')) {
    throw new Error('Failed to generate summary: model responded with tool use instead of a summary')
  }

  return summary
}

export type MessageTypeFilter = 'tools' | 'messages' | 'all'

/**
 * Returns true if the message matches the given type filter.
 * - 'tools': message contains at least one toolUseBlock or toolResultBlock
 * - 'messages': message contains no toolUseBlock or toolResultBlock
 * - 'all': always matches
 */
export function matchesMessageType(message: Message, filter: MessageTypeFilter): boolean {
  if (filter === 'all') return true
  const hasTool = message.content.some((b) => b.type === 'toolUseBlock' || b.type === 'toolResultBlock')
  if (filter === 'tools') return hasTool
  if (filter === 'messages') return !hasTool
  return false
}
