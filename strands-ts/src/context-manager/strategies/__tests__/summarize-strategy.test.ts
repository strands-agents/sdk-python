import { describe, it, expect, vi } from 'vitest'
import { Offload } from '../offload.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { StrategyContext } from '../../types.js'

vi.mock('../../../conversation-manager/compression/context-compression.js', () => ({
  adjustSplitPointForToolPairs: vi.fn((messages: Message[], splitPoint: number) => splitPoint),
  generateSummary: vi.fn(async () => new Message({ role: 'user', content: [new TextBlock('Summary of conversation')] })),
}))

function makeUserMessage(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function makeContext(messages: Message[], utilization = 0.5, model?: unknown): StrategyContext {
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

describe('Offload.summarize', () => {
  it('creates a strategy with correct name', () => {
    const strategy = Offload.summarize()
    expect(strategy.name).toBe('offload:summarize')
  })

  it('creates a strategy with .when() conditions', () => {
    const strategy = Offload.summarize({ ratio: 0.5 }).when({ utilization: 0.85 })
    expect(strategy.name).toBe('offload:summarize')
  })

  describe('apply', () => {
    it('returns false when no model is available', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const strategy = Offload.summarize()
      const context = makeContext(messages)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('returns false when not enough messages to summarize', async () => {
      const messages = [makeUserMessage('hello')]
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize({ preserveRecent: 10 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('summarizes oldest messages', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize({ ratio: 0.3, preserveRecent: 5 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      expect(messages).toHaveLength(15)
    })

    it('does not exceed preserveRecent', async () => {
      const messages = Array.from({ length: 12 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize({ ratio: 0.8, preserveRecent: 10 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      expect(messages).toHaveLength(11)
    })

    it('skips when utilization is below threshold', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize().when({ utilization: 0.85 })
      const context = makeContext(messages, 0.5, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
      expect(messages).toHaveLength(20)
    })

    it('fires when utilization exceeds threshold', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize({ preserveRecent: 5 }).when({ utilization: 0.85 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
    })

    it('fires unconditionally when no utilization threshold set', async () => {
      const messages = Array.from({ length: 20 }, (_, index) => makeUserMessage(`msg-${index}`))
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize({ preserveRecent: 5 })
      const context = makeContext(messages, 0.1, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
    })

    it('returns false on empty messages', async () => {
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize()
      const context = makeContext([], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })
  })
})
