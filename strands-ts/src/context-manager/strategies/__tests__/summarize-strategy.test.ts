import { describe, it, expect, vi } from 'vitest'
import { SummarizeStrategy } from '../summarize-strategy.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { PassContext } from '../../types.js'

vi.mock('../../../conversation-manager/compression/context-compression.js', () => ({
  adjustSplitPointForToolPairs: vi.fn((messages: Message[], splitPoint: number) => splitPoint),
  generateSummary: vi.fn(async () => new Message({ role: 'user', content: [new TextBlock('Summary of conversation')] })),
}))

function makeUserMessage(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function makeContext(messages: Message[], utilization = 0.5, model?: unknown): PassContext {
  const agent = createMockAgent({ messages })
  if (model) {
    ;(agent as unknown as Record<string, unknown>)['model'] = model
  }
  return {
    messages,
    agent,
    utilization,
    storage: new InMemoryStorage(),
  }
}

describe('SummarizeStrategy', () => {
  describe('constructor', () => {
    it('uses default config values', () => {
      const strategy = new SummarizeStrategy()
      expect(strategy.name).toBe('summarize')
      expect(strategy['_summaryRatio']).toBe(0.3)
      expect(strategy['_preserveRecent']).toBe(10)
      expect(strategy['_utilization']).toBeUndefined()
    })

    it('accepts custom config', () => {
      const strategy = new SummarizeStrategy({
        summaryRatio: 0.5,
        preserveRecent: 5,
        utilization: 0.85,
      })
      expect(strategy['_summaryRatio']).toBe(0.5)
      expect(strategy['_preserveRecent']).toBe(5)
      expect(strategy['_utilization']).toBe(0.85)
    })

    it('clamps ratio to min', () => {
      const strategy = new SummarizeStrategy({ summaryRatio: 0.01 })
      expect(strategy['_summaryRatio']).toBe(0.1)
    })

    it('clamps ratio to max', () => {
      const strategy = new SummarizeStrategy({ summaryRatio: 0.99 })
      expect(strategy['_summaryRatio']).toBe(0.8)
    })
  })

  describe('apply', () => {
    it('returns false when no model is available', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const strategy = new SummarizeStrategy()
      const context = makeContext(messages)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('returns false when not enough messages to summarize', async () => {
      const messages = [makeUserMessage('hello')]
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ preserveRecent: 10 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('summarizes oldest messages', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ summaryRatio: 0.3, preserveRecent: 5 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      // 20 * 0.3 = 6 messages summarized, replaced with 1
      expect(messages).toHaveLength(15)
    })

    it('does not exceed preserve_recent', async () => {
      const messages = Array.from({ length: 12 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ summaryRatio: 0.8, preserveRecent: 10 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      // min(floor(12*0.8)=9, 12-10=2) = 2 summarized, replaced with 1
      expect(messages).toHaveLength(11)
    })

    it('skips when utilization is below threshold', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ utilization: 0.85 })
      const context = makeContext(messages, 0.5, mockModel) // below 0.85

      const result = await strategy.apply(context)

      expect(result).toBe(false)
      expect(messages).toHaveLength(20) // unchanged
    })

    it('fires when utilization exceeds threshold', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ utilization: 0.85, preserveRecent: 5 })
      const context = makeContext(messages, 0.9, mockModel) // above 0.85

      const result = await strategy.apply(context)

      expect(result).toBe(true)
    })

    it('fires unconditionally when no utilization threshold set', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy({ preserveRecent: 5 })
      const context = makeContext(messages, 0.1, mockModel) // low utilization but no threshold

      const result = await strategy.apply(context)

      expect(result).toBe(true)
    })

    it('returns false on empty messages', async () => {
      const mockModel = { stream: vi.fn() }
      const strategy = new SummarizeStrategy()
      const context = makeContext([], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })
  })
})
