import { describe, it, expect } from 'vitest'
import { TruncateMethod } from '../methods/truncate-method.js'
import { Offload } from '../offload.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { StrategyContext } from '../../types.js'

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

function makeContext(messages: Message[], storage?: InMemoryStorage, utilization = 0.5): StrategyContext {
  return {
    messages,
    agent: createMockAgent({ messages }),
    utilization,
    storage: storage ?? new InMemoryStorage(),
  }
}

describe('TruncateMethod', () => {
  describe('constructor', () => {
    it('uses default config values', () => {
      const method = new TruncateMethod('toolResults')
      expect(method.name).toBe('offload:truncate')
    })

    it('accepts custom config via builder', () => {
      const strategy = Offload.truncate('toolResults', { previewTokens: 2000 })
        .when({ threshold: 5000, skipRecent: 5 })
      expect(strategy.name).toBe('offload:truncate')
    })
  })

  describe('apply', () => {
    it('offloads large tool results to storage', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText)]
      const storage = new InMemoryStorage()
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext(messages, storage)

      const result = await method.apply(context)

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
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext(messages, storage)

      const result = await method.apply(context)

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
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 3 })
      const context = makeContext(messages, storage)

      const result = await method.apply(context)

      expect(result).toBe(true)
      const firstBlock = messages[0]!.content[0] as ToolResultBlock
      expect((firstBlock.content[0] as TextBlock).text).toContain('[Offloaded:')
      const lastBlock = messages[3]!.content[0] as ToolResultBlock
      expect((lastBlock.content[0] as TextBlock).text).not.toContain('[Offloaded:')
    })

    it('skips error results when target is toolResults', async () => {
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
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext([message])

      const result = await method.apply(context)

      expect(result).toBe(false)
    })

    it('offloads error results when target is toolResultErrors', async () => {
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
      const method = new TruncateMethod('toolResultErrors', undefined, { skipRecent: 0 })
      const context = makeContext([message])

      const result = await method.apply(context)

      expect(result).toBe(true)
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
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext([message])

      const result = await method.apply(context)

      expect(result).toBe(false)
    })

    it('skips non-user messages', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const message = new Message({
        role: 'assistant',
        content: [new TextBlock(largeText)],
      })
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext([message])

      const result = await method.apply(context)

      expect(result).toBe(false)
    })

    it('builds head-tail preview', async () => {
      const largeText = 'HEAD'.repeat(1000) + 'MIDDLE'.repeat(5000) + 'TAIL'.repeat(1000)
      const messages = [makeToolResultMessage(largeText)]
      const storage = new InMemoryStorage()
      const method = new TruncateMethod('toolResults', { previewTokens: 100 }, { skipRecent: 0 })
      const context = makeContext(messages, storage)

      await method.apply(context)

      const block = messages[0]!.content[0] as ToolResultBlock
      const previewText = (block.content[0] as TextBlock).text
      expect(previewText).toContain('[...')
      expect(previewText).toContain('chars elided')
    })

    it('stores content to storage with correct key', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText, 'my-tool-id')]
      const storage = new InMemoryStorage()
      const method = new TruncateMethod('toolResults', undefined, { skipRecent: 0 })
      const context = makeContext(messages, storage)

      await method.apply(context)

      const keys = await storage.list('')
      expect(keys.some((key) => key.includes('offload/my-tool-id'))).toBe(true)
    })

    it('returns false for empty messages', async () => {
      const method = new TruncateMethod('toolResults')
      const context = makeContext([])

      const result = await method.apply(context)

      expect(result).toBe(false)
    })
  })

  describe('Offload builder', () => {
    it('Offload.truncate() creates a strategy', () => {
      const strategy = Offload.truncate('toolResults')
      expect(strategy.name).toBe('offload:truncate')
      expect(strategy.init).toBeDefined()
      expect(strategy.apply).toBeDefined()
    })

    it('Offload.truncate().when() creates a strategy with conditions', () => {
      const strategy = Offload.truncate('toolResults', { previewTokens: 500 })
        .when({ threshold: 1000, skipRecent: 5 })
      expect(strategy.name).toBe('offload:truncate')
    })

    it('Offload() shorthand creates a strategy', () => {
      const strategy = Offload('toolResults')
      expect(strategy.name).toBe('offload:truncate')
    })

    it('Offload.summarize() creates a strategy', () => {
      const strategy = Offload.summarize()
      expect(strategy.name).toBe('offload:summarize')
      expect(strategy.apply).toBeDefined()
    })

    it('Offload.summarize().when() creates a strategy with conditions', () => {
      const strategy = Offload.summarize({ ratio: 0.5 })
        .when({ utilization: 0.85 })
      expect(strategy.name).toBe('offload:summarize')
    })

    it('Offload.truncate() with tool name array targets specific tools', async () => {
      const largeText = 'x'.repeat(2500 * 4 + 100)
      const messages = [makeToolResultMessage(largeText, 'bash-tool-123')]
      const storage = new InMemoryStorage()
      const strategy = Offload.truncate(['bash']).when({ skipRecent: 0 })
      const context = makeContext(messages, storage)

      const result = await strategy.apply(context)

      expect(result).toBe(false)
    })
  })
})
