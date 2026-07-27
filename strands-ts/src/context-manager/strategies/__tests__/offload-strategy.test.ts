import { describe, it, expect } from 'vitest'
import { OffloadStrategy } from '../offload-strategy.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { PassContext } from '../../types.js'

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

function makeContext(messages: Message[], storage?: InMemoryStorage, utilization = 0.5): PassContext {
  return {
    messages,
    agent: createMockAgent({ messages }),
    utilization,
    storage: storage ?? new InMemoryStorage(),
  }
}

describe('OffloadStrategy', () => {
  describe('constructor', () => {
    it('uses default config values', () => {
      const strategy = new OffloadStrategy()
      expect(strategy.name).toBe('offload')
      expect(strategy['_maxResultTokens']).toBe(2500)
      expect(strategy['_previewTokens']).toBe(1000)
      expect(strategy['_skipRecent']).toBe(3)
    })

    it('accepts custom config', () => {
      const strategy = new OffloadStrategy({
        maxResultTokens: 5000,
        previewTokens: 2000,
        skipRecent: 5,
      })
      expect(strategy['_maxResultTokens']).toBe(5000)
      expect(strategy['_previewTokens']).toBe(2000)
      expect(strategy['_skipRecent']).toBe(5)
    })
  })

  describe('apply', () => {
    it('offloads large tool results to storage', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const storage = new InMemoryStorage()
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext(messages, storage)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      const content = messages[0]!.content[0]
      expect(content).toBeInstanceOf(ToolResultBlock)
      const block = content as ToolResultBlock
      expect(block.content[0]).toBeInstanceOf(TextBlock)
      expect((block.content[0] as TextBlock).text).toContain('[Offloaded:')
    })

    it('does not offload small results', async () => {
      const smallText = 'short result'
      const messages = [makeToolResultMessage(smallText)]
      const storage = new InMemoryStorage()
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext(messages, storage)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('skips recent messages', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [
        makeToolResultMessage(largeText, 'tool-1'),
        makeToolResultMessage(largeText, 'tool-2'),
        makeToolResultMessage(largeText, 'tool-3'),
        makeToolResultMessage(largeText, 'tool-4'),
      ]
      const storage = new InMemoryStorage()
      const strategy = new OffloadStrategy({ skipRecent: 3 })
      const context = makeContext(messages, storage)

      const result = await strategy.apply(context)

      expect(result).toBe(true)
      // Only the first message should be offloaded
      const firstBlock = messages[0]!.content[0] as ToolResultBlock
      expect((firstBlock.content[0] as TextBlock).text).toContain('[Offloaded:')
      // The last 3 should remain unchanged
      const lastBlock = messages[3]!.content[0] as ToolResultBlock
      expect((lastBlock.content[0] as TextBlock).text).not.toContain('[Offloaded:')
    })

    it('skips error results', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const message = new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'tool-err',
            status: 'error',
            content: [new TextBlock(largeText)],
          }),
        ],
      })
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext([message])

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('skips already offloaded results', async () => {
      const message = new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'tool-123',
            status: 'success',
            content: [new TextBlock('[Offloaded: 1 blocks, ~500 tokens]\npreview...')],
          }),
        ],
      })
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext([message])

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('skips non-user messages', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const message = new Message({
        role: 'assistant',
        content: [new TextBlock(largeText)],
      })
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext([message])

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })

    it('builds head-tail preview', async () => {
      const largeText = 'HEAD'.repeat(1000) + 'MIDDLE'.repeat(5000) + 'TAIL'.repeat(1000)
      const messages = [makeToolResultMessage(largeText)]
      const storage = new InMemoryStorage()
      const strategy = new OffloadStrategy({ skipRecent: 0, previewTokens: 100 })
      const context = makeContext(messages, storage)

      await strategy.apply(context)

      const block = messages[0]!.content[0] as ToolResultBlock
      const previewText = (block.content[0] as TextBlock).text
      expect(previewText).toContain('[...')
      expect(previewText).toContain('chars elided')
    })

    it('stores content to storage with correct key', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText, 'my-tool-id')]
      const storage = new InMemoryStorage()
      const strategy = new OffloadStrategy({ skipRecent: 0 })
      const context = makeContext(messages, storage)

      await strategy.apply(context)

      const keys = await storage.list('')
      expect(keys.some((key) => key.includes('offload/my-tool-id'))).toBe(true)
    })

    it('returns false for empty messages', async () => {
      const strategy = new OffloadStrategy()
      const context = makeContext([])

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })
  })
})
