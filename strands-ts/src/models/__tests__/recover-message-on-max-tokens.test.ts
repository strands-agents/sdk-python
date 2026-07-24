import { describe, expect, it } from 'vitest'
import { Message, TextBlock, ToolUseBlock } from '../../types/messages.js'
import { recoverMessageOnMaxTokensReached } from '../recover-message-on-max-tokens.js'

describe('recoverMessageOnMaxTokensReached', () => {
  it('replaces every tool use while preserving other message data', () => {
    const message = new Message({
      role: 'assistant',
      content: [
        new TextBlock('Working on it'),
        new ToolUseBlock({ name: 'calculator', toolUseId: 'tool-1', input: { expression: '2+2' } }),
        new ToolUseBlock({ name: '', toolUseId: 'tool-2', input: {} }),
      ],
      trackingId: '7b461221-eccb-4502-a949-ddf6c611705d',
      metadata: { custom: { source: 'test' } },
    })

    const recovered = recoverMessageOnMaxTokensReached(message)

    expect(recovered).toEqual(
      new Message({
        role: 'assistant',
        content: [
          new TextBlock('Working on it'),
          new TextBlock(
            "The selected tool calculator's tool use was incomplete due to maximum token limits being reached."
          ),
          new TextBlock(
            "The selected tool <unknown>'s tool use was incomplete due to maximum token limits being reached."
          ),
        ],
        trackingId: message.trackingId,
        metadata: { custom: { source: 'test' } },
      })
    )
  })
})
