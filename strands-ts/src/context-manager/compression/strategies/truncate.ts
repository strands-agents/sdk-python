import type { Message } from '../../../types/messages.js'
import { isProtected } from '../protection.js'
import { logger } from '../../../logging/logger.js'

export type TruncateOptions = {
  /** Positive: protect first N messages. Negative: protect last N messages. */
  protectFirst?: number
}

/**
 * Truncate oldest messages from the conversation, preserving tool use/result pairs.
 * Protected messages (by range) are never removed.
 *
 * @param messages - The messages array to mutate in place
 * @param windowSize - Maximum messages to keep
 * @param options - Options including protectFirst
 * @returns `true` if messages were removed, `false` if no valid trim point found
 */
export function truncate(messages: Message[], windowSize: number, options?: TruncateOptions): boolean {
  if (messages.length <= 2) return false

  const protectedRange = options?.protectFirst

  let trimIndex = messages.length <= windowSize ? 2 : messages.length - windowSize
  trimIndex = findValidTrimPoint(messages, trimIndex)

  if (trimIndex >= messages.length) {
    logger.warn(`window_size=<${windowSize}>, messages=<${messages.length}> | unable to trim, no valid trim point`)
    return false
  }

  // Collect non-protected indices in [0, trimIndex) to remove
  const indicesToRemove: number[] = []
  for (let i = 0; i < trimIndex; i++) {
    if (isProtected(messages, i, protectedRange)) continue
    indicesToRemove.push(i)
  }

  if (indicesToRemove.length === 0) {
    logger.warn(
      `window_size=<${windowSize}>, messages=<${messages.length}> | all messages in trim range are protected, unable to reduce`
    )
    return false
  }

  // Remove in reverse order to keep indices stable
  for (let i = indicesToRemove.length - 1; i >= 0; i--) {
    messages.splice(indicesToRemove[i]!, 1)
  }
  return true
}

/**
 * Find a valid trim point starting from the given index.
 * Skips positions that would leave orphaned toolResults or toolUse without a following toolResult.
 */
function findValidTrimPoint(messages: Message[], startIndex: number): number {
  let idx = startIndex
  while (idx < messages.length) {
    const msg = messages[idx]
    if (!msg) break

    if (msg.role !== 'user') {
      idx++
      continue
    }

    if (msg.content.some((b) => b.type === 'toolResultBlock')) {
      idx++
      continue
    }

    const hasToolUse = msg.content.some((b) => b.type === 'toolUseBlock')
    if (hasToolUse) {
      const next = messages[idx + 1]
      if (!next || !next.content.some((b) => b.type === 'toolResultBlock')) {
        idx++
        continue
      }
    }

    break
  }
  return idx
}
