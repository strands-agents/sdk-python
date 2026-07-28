import { describe, it, expect, vi } from 'vitest'
import { Offload } from '../offload.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { StrategyContext } from '../../types.js'

vi.mock('../methods/summarize.js', () => ({
  summarizeText: vi.fn(async (text: string) => `Summary of: ${text.slice(0, 20)}`),
}))

function makeToolResultMessage(text: string, toolUseId = 'tool-123'): Message {
  return new Message({
    role: 'user',
    content: [
      new ToolResultBlock({
        toolUseId,
        status: 'success',
        content: [new TextBlock(text)],
      }),
    ],
  })
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
  }
}

describe('Offload.summarize', () => {
  it('creates a strategy with correct name', () => {
    const strategy = Offload.summarize('toolResults')
    expect(strategy.name).toBe('offload:summarize')
  })

  it('creates a strategy with .when() conditions', () => {
    const strategy = Offload.summarize('toolResults').when({ utilization: 0.85 })
    expect(strategy.name).toBe('offload:summarize')
  })

  describe('apply', () => {
    it('returns false when no model is available', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const strategy = Offload.summarize('toolResults')
      const context = makeContext(messages)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('returns false when blocks are below threshold', async () => {
      const smallText = 'short result'
      const messages = [makeToolResultMessage(smallText)]
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults').when({ threshold: 2500 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('summarizes large tool results', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults')
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      const block = messages[0]!.content[0] as ToolResultBlock
      expect((block.content[0] as TextBlock).text).toContain('[Summarized:')
    })

    it('summarizes assistant text blocks', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const message = new Message({
        role: 'assistant',
        content: [new TextBlock(largeText)],
      })
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('assistantMessages')
      const context = makeContext([message], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      const block = message.content[0] as TextBlock
      expect(block.text).toContain('[Summarized:')
    })

    it('summarizes user text blocks', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const message = new Message({
        role: 'user',
        content: [new TextBlock(largeText)],
      })
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('userMessages')
      const context = makeContext([message], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      const block = message.content[0] as TextBlock
      expect(block.text).toContain('[Summarized:')
    })

    it('skips when utilization is below threshold', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults').when({ utilization: 0.85 })
      const context = makeContext(messages, 0.5, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('fires when utilization exceeds threshold', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults').when({ utilization: 0.85 })
      const context = makeContext(messages, 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
    })

    it('skips already summarized blocks', async () => {
      const message = new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'tool-123',
            status: 'success',
            content: [new TextBlock('[Summarized: ~500 tokens]\nsome summary')],
          }),
        ],
      })
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults')
      const context = makeContext([message], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('returns false on empty messages', async () => {
      const mockModel = { stream: vi.fn() }
      const strategy = Offload.summarize('toolResults')
      const context = makeContext([], 0.9, mockModel)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })
  })
})
