import { describe, it, expect, vi } from 'vitest'
import { Inject } from '../inject.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { StrategyContext } from '../../types.js'

vi.mock('../../../conversation-manager/compression/context-compression.js', () => ({
  adjustSplitPointForToolPairs: vi.fn((messages: Message[], splitPoint: number) => splitPoint),
  generateSummary: vi.fn(async () => new Message({ role: 'user', content: [new TextBlock('Summary of stash')] })),
}))

function makeContext(messages: Message[], storage: InMemoryStorage, utilization = 0.5, model?: unknown): StrategyContext {
  const agent = createMockAgent({ messages })
  if (model) {
    ;(agent as unknown as Record<string, unknown>)['model'] = model
  }
  return {
    messages,
    agent,
    utilization,
    storage,
  }
}

describe('Inject.truncate', () => {
  it('creates a strategy with correct name', () => {
    const strategy = Inject.truncate('stash')
    expect(strategy.name).toBe('inject:truncate')
  })

  it('creates a strategy with .when() conditions', () => {
    const strategy = Inject.truncate('stash', { previewTokens: 500 }).when({ utilization: 0.5 })
    expect(strategy.name).toBe('inject:truncate')
  })

  it('injects content from storage', async () => {
    const storage = new InMemoryStorage()
    await storage.write('stash/msg-1', new TextEncoder().encode('stored content'))
    const messages: Message[] = []
    const strategy = Inject.truncate('stash')
    const context = makeContext(messages, storage)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    expect(messages).toHaveLength(1)
    expect((messages[0]!.content[0] as TextBlock).text).toContain('stored content')
  })

  it('truncates large stored content', async () => {
    const storage = new InMemoryStorage()
    const largeContent = 'x'.repeat(50000)
    await storage.write('stash/msg-1', new TextEncoder().encode(largeContent))
    const messages: Message[] = []
    const strategy = Inject.truncate('stash', { previewTokens: 100 })
    const context = makeContext(messages, storage)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    expect(messages).toHaveLength(1)
    const text = (messages[0]!.content[0] as TextBlock).text
    expect(text).toContain('[... truncated ...]')
    expect(text.length).toBeLessThan(largeContent.length)
  })

  it('returns false when storage is empty', async () => {
    const storage = new InMemoryStorage()
    const messages: Message[] = []
    const strategy = Inject.truncate('stash')
    const context = makeContext(messages, storage)

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('skips when utilization is above threshold', async () => {
    const storage = new InMemoryStorage()
    await storage.write('stash/msg-1', new TextEncoder().encode('content'))
    const messages: Message[] = []
    const strategy = Inject.truncate('stash').when({ utilization: 0.5 })
    const context = makeContext(messages, storage, 0.8)

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('injects when utilization is below threshold', async () => {
    const storage = new InMemoryStorage()
    await storage.write('stash/msg-1', new TextEncoder().encode('content'))
    const messages: Message[] = []
    const strategy = Inject.truncate('stash').when({ utilization: 0.5 })
    const context = makeContext(messages, storage, 0.3)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
  })
})

describe('Inject.summarize', () => {
  it('creates a strategy with correct name', () => {
    const strategy = Inject.summarize('stash')
    expect(strategy.name).toBe('inject:summarize')
  })

  it('returns false when no model is available', async () => {
    const storage = new InMemoryStorage()
    await storage.write('stash/msg-1', new TextEncoder().encode('content'))
    const messages: Message[] = []
    const strategy = Inject.summarize('stash')
    const context = makeContext(messages, storage)

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('summarizes stored content and injects it', async () => {
    const storage = new InMemoryStorage()
    await storage.write('stash/msg-1', new TextEncoder().encode('first message'))
    await storage.write('stash/msg-2', new TextEncoder().encode('second message'))
    const messages: Message[] = []
    const mockModel = { stream: vi.fn() }
    const strategy = Inject.summarize('stash', { preserveRecent: 0 })
    const context = makeContext(messages, storage, 0.5, mockModel)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    expect(messages).toHaveLength(1)
  })

  it('returns false when storage is empty', async () => {
    const storage = new InMemoryStorage()
    const messages: Message[] = []
    const mockModel = { stream: vi.fn() }
    const strategy = Inject.summarize('stash')
    const context = makeContext(messages, storage, 0.5, mockModel)

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })
})

describe('Inject builder', () => {
  it('Inject() shorthand creates a truncate strategy', () => {
    const strategy = Inject('stash')
    expect(strategy.name).toBe('inject:truncate')
    expect(strategy.apply).toBeDefined()
  })
})
