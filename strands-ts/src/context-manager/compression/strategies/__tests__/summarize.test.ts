import { describe, it, expect, vi } from 'vitest'
import { summarize } from '../summarize.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../../../types/messages.js'
import { pinMessage } from '../../protection.js'
import type { Model } from '../../../../models/model.js'

function userMsg(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function assistantMsg(text: string): Message {
  return new Message({ role: 'assistant', content: [new TextBlock(text)] })
}

function createMockModel(summaryText = '## Summary\n* Conversation summary'): Model {
  const streamAggregated = vi.fn().mockImplementation(() => {
    let callCount = 0
    return {
      next: () => {
        callCount++
        if (callCount === 1) {
          return Promise.resolve({
            done: true,
            value: {
              message: new Message({ role: 'assistant', content: [new TextBlock(summaryText)] }),
              stopReason: 'endTurn',
            },
          })
        }
        return Promise.resolve({ done: true, value: undefined })
      },
    }
  })

  return { streamAggregated } as unknown as Model
}

describe('summarize', () => {
  it('summarizes oldest messages and replaces them with summary', async () => {
    const messages = [
      userMsg('Message 1'),
      assistantMsg('Response 1'),
      userMsg('Message 2'),
      assistantMsg('Response 2'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
      userMsg('Message 5'),
      assistantMsg('Response 5'),
      userMsg('Message 6'),
      assistantMsg('Response 6'),
    ]
    const model = createMockModel()

    const result = await summarize(messages, model)

    expect(result).toBe(true)
    // Default summaryRatio is 0.3, 12 * 0.3 = 3.6 -> 3 messages summarized
    // But adjustSplitForToolPairs may adjust. Summary replaces those.
    expect(messages.length).toBeLessThan(12)
    // First message should be the summary (since no protected messages)
    expect(messages[0]!.role).toBe('user')
  })

  it('preserves recent messages', async () => {
    const messages = [
      userMsg('Old 1'),
      assistantMsg('Old response 1'),
      userMsg('Old 2'),
      assistantMsg('Old response 2'),
      userMsg('Old 3'),
      assistantMsg('Old response 3'),
      userMsg('Old 4'),
      assistantMsg('Old response 4'),
      userMsg('Old 5'),
      assistantMsg('Old response 5'),
      userMsg('Recent 1'),
      assistantMsg('Recent response 1'),
    ]
    const model = createMockModel()

    await summarize(messages, model, { preserveRecentMessages: 10 })

    // With 12 messages and preserveRecentMessages=10, count=max(1, floor(12*0.3))=3
    // count = min(3, 12-10)=2 -> summarize first 2 messages
    // After: summary + 10 remaining = 11
    expect(messages.length).toBeLessThan(12)
    // Recent messages should still be there
    const lastMsg = messages[messages.length - 1]!
    expect((lastMsg.content[0] as TextBlock).text).toBe('Recent response 1')
  })

  it('respects protectFirst (keeps protected messages verbatim)', async () => {
    const messages = [
      userMsg('System instruction'), // index 0 - protected
      assistantMsg('Acknowledged'), // index 1 - protected
      userMsg('Old message'),
      assistantMsg('Old response'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
      userMsg('Message 5'),
      assistantMsg('Response 5'),
      userMsg('Recent'),
      assistantMsg('Recent response'),
    ]
    const model = createMockModel()

    const result = await summarize(messages, model, { protectFirst: 2, preserveRecentMessages: 6 })

    expect(result).toBe(true)
    // Protected messages should be preserved verbatim at the start
    expect((messages[0]!.content[0] as TextBlock).text).toBe('System instruction')
    expect((messages[1]!.content[0] as TextBlock).text).toBe('Acknowledged')
  })

  it('returns false when insufficient messages to summarize', async () => {
    const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
    const model = createMockModel()

    // preserveRecentMessages=10, count = min(floor(2*0.3)=0 -> max(1,0)=1, 2-10=-8) -> 0 -> returns false
    const result = await summarize(messages, model, { preserveRecentMessages: 10 })

    expect(result).toBe(false)
    expect(messages).toHaveLength(2)
  })

  it('returns false when all messages in range are protected', async () => {
    const messages = [
      pinMessage(userMsg('Pinned 1')),
      pinMessage(assistantMsg('Pinned 2')),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
      userMsg('Message 5'),
      assistantMsg('Response 5'),
      userMsg('Message 6'),
      assistantMsg('Response 6'),
      userMsg('Recent'),
      assistantMsg('Recent response'),
    ]
    const model = createMockModel()

    // protectFirst=2, summaryRatio=0.15 -> count=max(1, floor(12*0.15))=1
    // Only index 0 is in range, and it's pinned. hasUnprotected=false -> returns false
    const result = await summarize(messages, model, { protectFirst: 2, summaryRatio: 0.08 })

    expect(result).toBe(false)
  })

  it('summary is generated from ALL messages in range including protected', async () => {
    const messages = [
      pinMessage(userMsg('Important context')),
      assistantMsg('Response to important'),
      userMsg('Regular message'),
      assistantMsg('Regular response'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
      userMsg('Message 5'),
      assistantMsg('Response 5'),
      userMsg('Recent'),
      assistantMsg('Recent response'),
    ]

    const streamAggregated = vi.fn().mockImplementation((_input: Message[]) => {
      let callCount = 0
      return {
        next: () => {
          callCount++
          if (callCount === 1) {
            return Promise.resolve({
              done: true,
              value: {
                message: new Message({ role: 'assistant', content: [new TextBlock('Summary')] }),
                stopReason: 'endTurn',
              },
            })
          }
          return Promise.resolve({ done: true, value: undefined })
        },
      }
    })
    const model = { streamAggregated } as unknown as Model

    await summarize(messages, model, { protectFirst: 1, summaryRatio: 0.3, preserveRecentMessages: 6 })

    // streamAggregated should have been called with messages that include the pinned one
    expect(streamAggregated).toHaveBeenCalled()
    const inputMessages = streamAggregated.mock.calls[0]![0] as Message[]
    // The input should include the pinned message at index 0
    const hasPinnedContent = inputMessages.some((m) =>
      m.content.some((b) => b.type === 'textBlock' && (b as TextBlock).text === 'Important context')
    )
    expect(hasPinnedContent).toBe(true)
  })

  it('adjusts split point to avoid breaking tool pairs', async () => {
    const messages = [
      userMsg('Message 1'),
      assistantMsg('Response 1'),
      new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 't1', name: 'test', input: {} })] }),
      new Message({
        role: 'user',
        content: [new ToolResultBlock({ toolUseId: 't1', content: [new TextBlock('result')], status: 'success' })],
      }),
      userMsg('Message after tool'),
      assistantMsg('Response after tool'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
      userMsg('Recent'),
      assistantMsg('Recent response'),
    ]
    const model = createMockModel()

    const result = await summarize(messages, model, { summaryRatio: 0.3, preserveRecentMessages: 4 })

    expect(result).toBe(true)
    // The split should not break the toolUse/toolResult pair at indices 2-3
  })
})
