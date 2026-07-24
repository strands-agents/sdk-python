import { Message, TextBlock } from '../types/messages.js'

/**
 * Replace tool uses in a token-limited response with resumable text.
 *
 * A tool use may be incomplete when generation reaches the token limit, so it
 * must not remain executable in conversation history. Non-tool content and
 * message identity are preserved.
 *
 * @param message - Partial model response produced before the token limit.
 * @returns A message safe to append to conversation history.
 * @internal
 */
export function recoverMessageOnMaxTokensReached(message: Message): Message {
  const content = message.content.map((block) => {
    if (block.type !== 'toolUseBlock') {
      return block
    }

    const toolName = block.name || '<unknown>'
    return new TextBlock(
      `The selected tool ${toolName}'s tool use was incomplete due to maximum token limits being reached.`
    )
  })

  return new Message({
    role: message.role,
    content,
    trackingId: message.trackingId,
    ...(message.metadata !== undefined && { metadata: message.metadata }),
  })
}
