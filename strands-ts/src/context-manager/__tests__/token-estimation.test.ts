import { describe, it, expect, vi } from 'vitest'
import { estimateInputTokens } from '../token-estimation.js'
import { Message, TextBlock } from '../../types/messages.js'
import type { Model } from '../../models/model.js'

function userMsg(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function assistantMsg(
  text: string,
  usage?: { inputTokens: number; outputTokens: number; totalTokens: number }
): Message {
  return new Message({
    role: 'assistant',
    content: [new TextBlock(text)],
    ...(usage && { metadata: { usage } }),
  })
}

function mockModel(countTokens?: (messages: Message[]) => Promise<number>): Model {
  return {
    countTokens: countTokens ?? vi.fn().mockResolvedValue(100),
  } as unknown as Model
}

describe('estimateInputTokens', () => {
  it('returns baseline from last assistant message usage metadata', async () => {
    const messages = [
      userMsg('hello'),
      assistantMsg('response', { inputTokens: 50, outputTokens: 20, totalTokens: 70 }),
    ]
    const model = mockModel()

    const result = await estimateInputTokens(messages, model)

    expect(result).toBe(70) // 50 + 20
  })

  it('adds new message tokens to baseline when messages exist after the assistant message', async () => {
    const messages = [
      userMsg('hello'),
      assistantMsg('response', { inputTokens: 50, outputTokens: 20, totalTokens: 70 }),
      userMsg('follow up'),
    ]
    const countTokens = vi.fn().mockResolvedValue(15)
    const model = mockModel(countTokens)

    const result = await estimateInputTokens(messages, model)

    expect(result).toBe(85) // 70 + 15
    expect(countTokens).toHaveBeenCalledWith([messages[2]])
  })

  it('uses the last assistant message with usage (not earlier ones)', async () => {
    const messages = [
      userMsg('hello'),
      assistantMsg('first', { inputTokens: 10, outputTokens: 5, totalTokens: 15 }),
      userMsg('second'),
      assistantMsg('latest', { inputTokens: 80, outputTokens: 30, totalTokens: 110 }),
    ]
    const model = mockModel()

    const result = await estimateInputTokens(messages, model)

    expect(result).toBe(110) // 80 + 30
  })

  it('falls back to model.countTokens when no assistant message has usage metadata', async () => {
    const messages = [
      userMsg('hello'),
      new Message({ role: 'assistant', content: [new TextBlock('no metadata')] }),
      userMsg('world'),
    ]
    const countTokens = vi.fn().mockResolvedValue(42)
    const model = mockModel(countTokens)

    const result = await estimateInputTokens(messages, model)

    expect(result).toBe(42)
    expect(countTokens).toHaveBeenCalledWith(messages)
  })

  it('falls back to model.countTokens when there are no assistant messages', async () => {
    const messages = [userMsg('hello')]
    const countTokens = vi.fn().mockResolvedValue(10)
    const model = mockModel(countTokens)

    const result = await estimateInputTokens(messages, model)

    expect(result).toBe(10)
    expect(countTokens).toHaveBeenCalledWith(messages)
  })

  it('returns undefined on error', async () => {
    const messages = [userMsg('hello')]
    const countTokens = vi.fn().mockRejectedValue(new Error('API error'))
    const model = mockModel(countTokens)

    const result = await estimateInputTokens(messages, model)

    expect(result).toBeUndefined()
  })
})
