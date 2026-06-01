import { z } from 'zod'
import { Message, type ToolUseBlock, type ToolResultBlock } from '../../types/messages.js'
import { tool } from '../../tools/tool-factory.js'

// --- Pin utilities ---

/**
 * Check if a single message is pinned.
 *
 * @param message - The message to check
 * @returns `true` if the message has `metadata.custom.pinned === true`
 */
export function isPinned(message: Message): boolean
/**
 * Check if a message is pinned, including tool-pair partner protection.
 * Returns `true` if the message at `index` is pinned, or if it is the
 * adjacent tool-pair partner (toolUse/toolResult) of a pinned message,
 * matched by toolUseId.
 *
 * @param messages - The full messages array
 * @param index - The index to check
 * @returns `true` if the message or its tool-pair partner is pinned
 */
export function isPinned(messages: Message[], index: number): boolean
export function isPinned(messageOrMessages: Message | Message[], index?: number): boolean {
  if (index === undefined) {
    return (messageOrMessages as Message).metadata?.custom?.pinned === true
  }

  const messages = messageOrMessages as Message[]
  const msg = messages[index]!
  if (msg.metadata?.custom?.pinned === true) return true

  const toolResultBlocks = msg.content.filter((b): b is ToolResultBlock => b.type === 'toolResultBlock')
  if (toolResultBlocks.length > 0 && index > 0) {
    const prev = messages[index - 1]!
    if (prev.metadata?.custom?.pinned === true) {
      const resultIds = new Set(toolResultBlocks.map((b) => b.toolUseId))
      if (prev.content.some((b) => b.type === 'toolUseBlock' && resultIds.has((b as ToolUseBlock).toolUseId))) {
        return true
      }
    }
  }

  const toolUseBlocks = msg.content.filter((b): b is ToolUseBlock => b.type === 'toolUseBlock')
  if (toolUseBlocks.length > 0 && index + 1 < messages.length) {
    const next = messages[index + 1]!
    if (next.metadata?.custom?.pinned === true) {
      const useIds = new Set(toolUseBlocks.map((b) => b.toolUseId))
      if (next.content.some((b) => b.type === 'toolResultBlock' && useIds.has((b as ToolResultBlock).toolUseId))) {
        return true
      }
    }
  }

  return false
}

/**
 * Returns a new Message marked as pinned (protected from eviction during context reduction).
 */
export function pinMessage(message: Message): Message {
  return new Message({
    role: message.role,
    content: message.content,
    metadata: {
      ...message.metadata,
      custom: { ...message.metadata?.custom, pinned: true },
    },
  })
}

/**
 * Returns a new Message with pinning removed.
 */
export function unpinMessage(message: Message): Message {
  const { pinned: _, ...restCustom } = message.metadata?.custom ?? {}
  const { custom: __, ...restMetadata } = message.metadata ?? {}
  const hasCustom = Object.keys(restCustom).length > 0
  const hasMetadata = hasCustom || Object.keys(restMetadata).length > 0
  const metadata = hasMetadata ? { ...restMetadata, ...(hasCustom ? { custom: restCustom } : {}) } : undefined

  return new Message({
    role: message.role,
    content: message.content,
    ...(metadata !== undefined ? { metadata } : {}),
  })
}

/**
 * Agent-invokable tool that pins or unpins a message in the conversation history.
 */
export const pinMessageTool = tool({
  name: 'pin_message',
  description:
    'Pin or unpin a message in the conversation history. ' +
    'Pinned messages are protected from eviction during context reduction. ' +
    'Use this to preserve important context that should not be summarized or trimmed away.',
  inputSchema: z.object({
    index: z.number().int().min(0).describe('The zero-based index of the message in the conversation history.'),
    action: z.enum(['pin', 'unpin']).default('pin').describe('Whether to pin or unpin the message.'),
  }),
  callback: ({ index, action }, context) => {
    const messages = context!.agent.messages
    if (index >= messages.length) {
      return `Invalid index ${index}. Conversation has ${messages.length} messages (indices 0-${messages.length - 1}).`
    }
    messages[index] = action === 'pin' ? pinMessage(messages[index]!) : unpinMessage(messages[index]!)
    return `${action === 'pin' ? 'Pinned' : 'Unpinned'} message at index ${index}.`
  },
})

// --- Protection check ---

/**
 * Check if a message at the given index is protected from eviction.
 * A message is protected if it is pinned, within the protectFirst range,
 * or is a tool-pair partner of a protected message.
 */
export function isProtected(messages: Message[], index: number, protectFirst?: number): boolean {
  if (isPinned(messages, index)) return true
  if (protectFirst === undefined || protectFirst <= 0) return false
  if (index < protectFirst) return true

  // Tool-pair partner: protect a toolResult just outside the range if its
  // preceding toolUse is inside the range (prevents orphaning)
  const msg = messages[index]!
  if (msg.content.some((b) => b.type === 'toolResultBlock') && index > 0 && index - 1 < protectFirst) {
    const prev = messages[index - 1]!
    if (prev.content.some((b) => b.type === 'toolUseBlock')) return true
  }

  return false
}
