import { describe, it, expect, vi } from 'vitest'
import { summarizeContextTool, truncateContextTool, pinTool } from '../agentic/agentic-context.js'
import { pinMessage } from '../compression/pin-message.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../types/messages.js'

function textMsg(role: 'user' | 'assistant', text: string): Message {
  return new Message({ role, content: [new TextBlock(text)] })
}

function makeMessages(count: number): Message[] {
  return Array.from({ length: count }, (_, i) => textMsg(i % 2 === 0 ? 'user' : 'assistant', `Message ${i + 1}`))
}

function mockModel(summaryText = 'Summary of older messages') {
  const message = new Message({ role: 'assistant', content: [new TextBlock(summaryText)] })
  return {
    streamAggregated: vi.fn(() => ({
      next: vi
        .fn()
        .mockResolvedValueOnce({ done: false, value: undefined })
        .mockResolvedValueOnce({ done: true, value: { message } }),
      [Symbol.asyncIterator]: vi.fn(),
    })),
  }
}

function makeContext(messages: Message[], model?: any) {
  return { agent: { messages, model: model ?? {} } } as any
}

describe('summarizeContextTool', () => {
  it('returns message when not enough messages to summarize', async () => {
    const messages = makeMessages(4)
    const result = await summarizeContextTool.invoke({ keepRecent: 10 }, makeContext(messages))
    expect(result).toContain('No summarization performed')
    expect(result).toContain('not enough eligible messages')
    expect(messages).toHaveLength(4)
  })

  it('summarizes eligible messages and splices the array', async () => {
    const model = mockModel('Summary')
    const messages = makeMessages(20)
    const result = await summarizeContextTool.invoke(
      { keepRecent: 10, summaryRatio: 0.5 },
      makeContext(messages, model)
    )
    expect(result).toContain('Summarized')
    expect(result).toContain('message(s)')
    expect(messages.length).toBeLessThan(20)
    expect(messages[0]!.role).toBe('user')
  })

  it('preserves pinned messages during summarization', async () => {
    const model = mockModel('Summary')
    const messages = makeMessages(20)
    pinMessage(messages, 1)
    const pinnedText = (messages[1]!.content[0] as TextBlock).text

    await summarizeContextTool.invoke({ keepRecent: 6, summaryRatio: 0.5 }, makeContext(messages, model))

    const texts = messages.map((m) => (m.content[0] as TextBlock).text)
    expect(texts).toContain(pinnedText)
  })

  it('respects messageType filter', async () => {
    const model = mockModel('Summary')
    const messages: Message[] = [
      new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'id-1', name: 'tool1', input: {} })] }),
      new Message({
        role: 'user',
        content: [new ToolResultBlock({ toolUseId: 'id-1', status: 'success', content: [new TextBlock('Result')] })],
      }),
      ...makeMessages(14),
    ]
    const result = await summarizeContextTool.invoke(
      { keepRecent: 10, summaryRatio: 0.3, messageType: 'messages' },
      makeContext(messages, model)
    )
    expect(result).toContain('"messages"')
  })

  it('returns message when no eligible messages in range after filtering', async () => {
    const messages: Message[] = [
      new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'id-1', name: 't', input: {} })] }),
      new Message({
        role: 'user',
        content: [new ToolResultBlock({ toolUseId: 'id-1', status: 'success', content: [new TextBlock('R')] })],
      }),
      new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'id-2', name: 't', input: {} })] }),
      new Message({
        role: 'user',
        content: [new ToolResultBlock({ toolUseId: 'id-2', status: 'success', content: [new TextBlock('R')] })],
      }),
      ...makeMessages(12),
    ]
    const result = await summarizeContextTool.invoke(
      { keepRecent: 12, summaryRatio: 0.3, messageType: 'messages' },
      makeContext(messages, mockModel())
    )
    expect(result).toContain('No summarization performed')
  })

  it('returns failure message when model throws', async () => {
    const model = {
      streamAggregated: vi.fn(() => ({
        next: vi.fn().mockRejectedValueOnce(new Error('model error')),
        [Symbol.asyncIterator]: vi.fn(),
      })),
    }
    const messages = makeMessages(20)
    const result = await summarizeContextTool.invoke({ keepRecent: 5, summaryRatio: 0.5 }, makeContext(messages, model))
    expect(result).toContain('Summarization failed')
  })
})

describe('truncateContextTool', () => {
  it('returns message when conversation is too short', async () => {
    const messages = makeMessages(2)
    const result = await truncateContextTool.invoke({}, makeContext(messages))
    expect(result).toContain('No messages dropped')
    expect(result).toContain('only has 2 messages')
    expect(messages).toHaveLength(2)
  })

  it('drops oldest messages respecting window', async () => {
    const messages = makeMessages(20)
    const result = await truncateContextTool.invoke({ keepRecent: 5 }, makeContext(messages))
    expect(result).toContain('Dropped')
    expect(result).toContain('remaining')
    expect(messages.length).toBeLessThan(20)
    expect(messages.length).toBeGreaterThanOrEqual(4)
    const lastMsg = messages[messages.length - 1]!
    expect((lastMsg.content[0] as TextBlock).text).toBe('Message 20')
  })

  it('preserves pinned messages during truncation', async () => {
    const messages = makeMessages(20)
    pinMessage(messages, 0)
    pinMessage(messages, 1)
    const pinnedText0 = (messages[0]!.content[0] as TextBlock).text
    const pinnedText1 = (messages[1]!.content[0] as TextBlock).text

    await truncateContextTool.invoke({ keepRecent: 5 }, makeContext(messages))

    const remainingTexts = messages.map((m) => (m.content[0] as TextBlock).text)
    expect(remainingTexts).toContain(pinnedText0)
    expect(remainingTexts).toContain(pinnedText1)
  })

  it('respects messageType filter', async () => {
    const messages: Message[] = [
      textMsg('user', 'Hello'),
      textMsg('assistant', 'Hi'),
      new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId: 'id-1', name: 't', input: {} })] }),
      new Message({
        role: 'user',
        content: [new ToolResultBlock({ toolUseId: 'id-1', status: 'success', content: [new TextBlock('R')] })],
      }),
      ...makeMessages(10),
    ]
    const result = await truncateContextTool.invoke({ keepRecent: 5, messageType: 'messages' }, makeContext(messages))
    expect(result).toContain('"messages"')
  })

  it('returns message when no valid trim point found', async () => {
    const messages = Array.from(
      { length: 6 },
      (_, i) =>
        new Message({
          role: 'user',
          content: [new ToolResultBlock({ toolUseId: `id-${i}`, status: 'success', content: [new TextBlock(`R`)] })],
        })
    )
    const result = await truncateContextTool.invoke({ keepRecent: 2 }, makeContext(messages))
    expect(result).toContain('No messages dropped')
    expect(result).toContain('no valid trim point')
  })
})

describe('pinTool', () => {
  describe('current_turn selector', () => {
    it('pins last user+assistant exchange', async () => {
      const messages = [
        textMsg('user', 'First question'),
        textMsg('assistant', 'First answer'),
        textMsg('user', 'Second question'),
        textMsg('assistant', 'Second answer'),
      ]
      const result = await pinTool.invoke(
        { selector: { type: 'current_turn' as const }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('Pinned')
      expect(result).toContain('2 message(s)')
      expect(messages[2]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[3]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[0]!.metadata?.custom?.pinned).toBeUndefined()
    })

    it('pins trailing user messages when last is user role', async () => {
      const messages = [textMsg('user', 'First'), textMsg('assistant', 'Answer'), textMsg('user', 'Follow-up')]
      await pinTool.invoke(
        { selector: { type: 'current_turn' as const }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(messages[2]!.metadata?.custom?.pinned).toBe(true)
    })
  })

  describe('last_n selector', () => {
    it('pins the last N messages', async () => {
      const messages = makeMessages(6)
      const result = await pinTool.invoke(
        { selector: { type: 'last_n' as const, count: 3 }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('Pinned')
      expect(result).toContain('3 message(s)')
      expect(messages[3]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[4]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[5]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[0]!.metadata?.custom?.pinned).toBeUndefined()
    })

    it('clamps count to conversation length', async () => {
      const messages = makeMessages(3)
      const result = await pinTool.invoke(
        { selector: { type: 'last_n' as const, count: 100 }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('3 message(s)')
    })
  })

  describe('indices selector', () => {
    it('pins messages at specific indices', async () => {
      const messages = makeMessages(6)
      const result = await pinTool.invoke(
        { selector: { type: 'indices' as const, indices: [0, 2, 4] }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('Pinned')
      expect(result).toContain('3 message(s)')
      expect(messages[0]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[1]!.metadata?.custom?.pinned).toBeUndefined()
      expect(messages[2]!.metadata?.custom?.pinned).toBe(true)
      expect(messages[4]!.metadata?.custom?.pinned).toBe(true)
    })

    it('returns error when all indices out of range', async () => {
      const messages = makeMessages(3)
      const result = await pinTool.invoke(
        { selector: { type: 'indices' as const, indices: [10, 20, 30] }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('All indices out of range')
      expect(result).toContain('3 messages')
    })

    it('filters out-of-range indices but pins valid ones', async () => {
      const messages = makeMessages(4)
      const result = await pinTool.invoke(
        { selector: { type: 'indices' as const, indices: [1, 99] }, action: 'pin' as const },
        makeContext(messages)
      )
      expect(result).toContain('Pinned')
      expect(result).toContain('1 message(s)')
      expect(messages[1]!.metadata?.custom?.pinned).toBe(true)
    })
  })

  describe('unpin action', () => {
    it('removes pin from previously pinned messages', async () => {
      const messages = makeMessages(4)
      pinMessage(messages, 1)
      pinMessage(messages, 2)

      const result = await pinTool.invoke(
        { selector: { type: 'indices' as const, indices: [1, 2] }, action: 'unpin' as const },
        makeContext(messages)
      )
      expect(result).toContain('Unpinned')
      expect(result).toContain('2 message(s)')
      expect(messages[1]!.metadata?.custom?.pinned).toBeUndefined()
      expect(messages[2]!.metadata?.custom?.pinned).toBeUndefined()
    })

    it('unpins with last_n selector', async () => {
      const messages = makeMessages(4)
      pinMessage(messages, 2)
      pinMessage(messages, 3)

      await pinTool.invoke(
        { selector: { type: 'last_n' as const, count: 2 }, action: 'unpin' as const },
        makeContext(messages)
      )
      expect(messages[2]!.metadata?.custom?.pinned).toBeUndefined()
      expect(messages[3]!.metadata?.custom?.pinned).toBeUndefined()
    })
  })

  describe('empty conversation', () => {
    it('returns appropriate message', async () => {
      const result = await pinTool.invoke(
        { selector: { type: 'current_turn' as const }, action: 'pin' as const },
        makeContext([])
      )
      expect(result).toBe('No messages in the conversation.')
    })
  })
})
