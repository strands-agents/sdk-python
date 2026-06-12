import { z } from 'zod'
import { Message } from '../../types/messages.js'
import { tool } from '../../tools/tool-factory.js'
import { pinMessage, unpinMessage, isPinned } from '../compression/pin-message.js'
import {
  generateSummary,
  adjustSplitPointForToolPairs,
  findValidTrimPoint,
  matchesMessageType,
  type MessageTypeFilter,
} from '../compression/context-compression.js'

const DEFAULT_KEEP_RECENT = 10
const DEFAULT_SUMMARY_RATIO = 0.3
const MIN_SUMMARY_RATIO = 0.1
const MAX_SUMMARY_RATIO = 0.8
const MIN_MESSAGES_FOR_OPERATION = 2

const messageTypeSchema = z
  .enum(['tools', 'messages', 'all'])
  .optional()
  .describe(
    'Filter which messages to target. "tools" targets only tool use/result messages, ' +
      '"messages" targets only non-tool messages, "all" (default) targets everything.'
  )

/**
 * Partition messages in [0, rangeEnd) into three buckets:
 * - pinned: protected from eviction (includes tool-pair partners)
 * - eligible: unpinned and matching the filter (will be evicted)
 * - skipped: unpinned but not matching the filter (preserved in place)
 */
function partitionByFilter(
  messages: Message[],
  rangeEnd: number,
  filter: MessageTypeFilter
): { pinned: Message[]; eligible: Message[]; skipped: Message[] } {
  const pinned: Message[] = []
  const eligible: Message[] = []
  const skipped: Message[] = []

  for (let i = 0; i < rangeEnd; i++) {
    const msg = messages[i]!
    if (isPinned(messages, i)) {
      pinned.push(msg)
    } else if (matchesMessageType(msg, filter)) {
      eligible.push(msg)
    } else {
      skipped.push(msg)
    }
  }

  return { pinned, eligible, skipped }
}

export const summarizeContextTool = tool({
  name: 'summarize_context',
  description:
    'Compress the oldest messages in your conversation into a concise summary to free up context space. ' +
    'The summary preserves key information while reducing token usage. ' +
    'Recent messages are kept verbatim. Pinned messages are never summarized away. ' +
    'Often most useful with messageType "messages" to preserve tool results verbatim while condensing discussion.',
  inputSchema: z.object({
    keepRecent: z
      .number()
      .int()
      .min(MIN_MESSAGES_FOR_OPERATION)
      .optional()
      .describe(`Minimum number of recent messages to preserve verbatim. Defaults to ${DEFAULT_KEEP_RECENT}.`),
    summaryRatio: z
      .number()
      .min(MIN_SUMMARY_RATIO)
      .max(MAX_SUMMARY_RATIO)
      .optional()
      .describe(
        `Fraction of the oldest messages to fold into the summary (${MIN_SUMMARY_RATIO}–${MAX_SUMMARY_RATIO}). Defaults to ${DEFAULT_SUMMARY_RATIO}.`
      ),
    messageType: messageTypeSchema,
  }),
  callback: async ({ keepRecent, summaryRatio, messageType }, context) => {
    const agent = context!.agent
    const messages = agent.messages
    const before = messages.length
    const filter: MessageTypeFilter = messageType ?? 'all'
    const preserveRecent = keepRecent ?? DEFAULT_KEEP_RECENT
    const ratio = Math.max(MIN_SUMMARY_RATIO, Math.min(MAX_SUMMARY_RATIO, summaryRatio ?? DEFAULT_SUMMARY_RATIO))

    let splitPoint = Math.max(1, Math.floor(messages.length * ratio))
    splitPoint = Math.min(splitPoint, messages.length - preserveRecent)
    if (splitPoint <= 0) {
      return `No summarization performed: not enough eligible messages to compress (conversation has ${before} messages, preserving recent ${preserveRecent}).`
    }

    splitPoint = adjustSplitPointForToolPairs(messages, splitPoint)

    const { pinned, eligible, skipped } = partitionByFilter(messages, splitPoint, filter)

    if (eligible.length === 0) {
      return `No summarization performed: no ${filter === 'all' ? 'eligible' : `"${filter}"`} messages found in range (conversation has ${before} messages).`
    }

    let summaryMessage
    try {
      summaryMessage = await generateSummary(eligible, agent.model)
    } catch {
      return 'Summarization failed: no response from model.'
    }

    messages.splice(0, splitPoint, ...pinned, ...skipped, summaryMessage)

    const removed = before - messages.length
    return `Summarized ${eligible.length} ${filter === 'all' ? '' : `"${filter}" `}message(s). Removed ${removed} message(s), ${messages.length} remaining.`
  },
})

export const truncateContextTool = tool({
  name: 'truncate_context',
  description:
    'Drop the oldest messages from your conversation history entirely to free up context space. ' +
    'Use this when older messages are no longer relevant and do not need to be preserved in any form. ' +
    'Pinned messages are always kept. Tool-call pairs are preserved together. ' +
    'Often most useful with messageType "tools" since tool results tend to be large and lose relevance quickly.',
  inputSchema: z.object({
    keepRecent: z
      .number()
      .int()
      .min(MIN_MESSAGES_FOR_OPERATION)
      .optional()
      .describe(`Number of most recent messages to keep. Everything older (and unpinned) is dropped. Defaults to ${DEFAULT_KEEP_RECENT}.`),
    messageType: messageTypeSchema,
  }),
  callback: ({ keepRecent, messageType }, context) => {
    const agent = context!.agent
    const messages = agent.messages
    const before = messages.length
    const filter: MessageTypeFilter = messageType ?? 'all'
    const windowSize = keepRecent ?? DEFAULT_KEEP_RECENT

    if (messages.length <= MIN_MESSAGES_FOR_OPERATION) {
      return `No messages dropped: conversation only has ${before} messages.`
    }

    const startIndex = messages.length <= windowSize ? MIN_MESSAGES_FOR_OPERATION : messages.length - windowSize
    const trimPoint = findValidTrimPoint(messages, startIndex)

    if (trimPoint >= messages.length) {
      return `No messages dropped: no valid trim point found (conversation has ${before} messages).`
    }

    const { pinned, eligible, skipped } = partitionByFilter(messages, trimPoint, filter)

    if (eligible.length === 0) {
      return `No messages dropped: no ${filter === 'all' ? 'eligible' : `"${filter}"`} messages found in range (conversation has ${before} messages).`
    }

    messages.splice(0, trimPoint, ...pinned, ...skipped)

    const dropped = before - messages.length
    return `Dropped ${dropped} ${filter === 'all' ? '' : `"${filter}" `}message(s). ${messages.length} remaining.`
  },
})

export const pinTool = tool({
  name: 'pin',
  description:
    'Pin or unpin messages in the conversation history. ' +
    'Pinned messages are protected from eviction during context reduction (summarize or truncate). ' +
    'Use this to preserve important context that should not be lost. ' +
    'Select messages using relative references: pin the current exchange, the last N messages, or specific indices.',
  inputSchema: z.object({
    selector: z
      .discriminatedUnion('type', [
        z.object({
          type: z.literal('current_turn'),
        }),
        z.object({
          type: z.literal('last_n'),
          count: z.number().int().min(1).describe('Number of messages from the end to select.'),
        }),
        z.object({
          type: z.literal('indices'),
          indices: z.array(z.number().int().min(0)).min(1).describe('Zero-based message indices to select.'),
        }),
      ])
      .describe(
        'How to select messages. "current_turn" pins the last user+assistant exchange. ' +
          '"last_n" pins the N most recent messages. ' +
          '"indices" pins specific messages by position.'
      ),
    action: z.enum(['pin', 'unpin']).default('pin').describe('Whether to pin or unpin the selected messages.'),
  }),
  callback: ({ selector, action }, context) => {
    const messages = context!.agent.messages

    if (messages.length === 0) {
      return 'No messages in the conversation.'
    }

    let targetIndices: number[]

    if (selector.type === 'current_turn') {
      targetIndices = []
      let i = messages.length - 1

      while (i >= 0 && messages[i]!.role === 'assistant') {
        targetIndices.push(i)
        i--
      }
      while (i >= 0 && messages[i]!.role === 'user') {
        targetIndices.push(i)
        i--
      }

      if (targetIndices.length === 0) {
        return 'Could not identify the current turn.'
      }
    } else if (selector.type === 'last_n') {
      const count = Math.min(selector.count, messages.length)
      targetIndices = Array.from({ length: count }, (_, k) => messages.length - 1 - k)
    } else {
      targetIndices = selector.indices.filter((i) => i < messages.length)
      const outOfRange = selector.indices.filter((i) => i >= messages.length)
      if (outOfRange.length > 0 && targetIndices.length === 0) {
        return `All indices out of range (conversation has ${messages.length} messages).`
      }
    }

    for (const index of targetIndices) {
      if (action === 'pin') {
        pinMessage(messages, index)
      } else {
        unpinMessage(messages, index)
      }
    }

    const verb = action === 'pin' ? 'Pinned' : 'Unpinned'
    return `${verb} ${targetIndices.length} message(s) (indices ${targetIndices.sort((a, b) => a - b).join(', ')}).`
  },
})
