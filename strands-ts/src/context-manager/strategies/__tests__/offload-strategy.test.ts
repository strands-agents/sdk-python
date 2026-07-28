import { describe, it, expect } from 'vitest'
import { Offload } from '../offload.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../../types/messages.js'
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

function makeContext(messages: Message[], utilization = 0.5): StrategyContext {
  return {
    messages,
    agent: createMockAgent({ messages }),
    utilization,
  }
}

describe('Offload.truncate', () => {
  it('creates a strategy with correct name', () => {
    const strategy = Offload.truncate('toolResults')
    expect(strategy.name).toBe('offload:truncate')
  })

  it('creates a strategy with .when() conditions', () => {
    const strategy = Offload.truncate('toolResults', { previewTokens: 500 })
      .when({ threshold: 1000 })
    expect(strategy.name).toBe('offload:truncate')
  })

  it('truncates large tool results', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const messages = [makeToolResultMessage(largeText)]
    const strategy = Offload.truncate('toolResults')
    const context = makeContext(messages)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const content = messages[0]!.content[0]
    expect(content).toBeInstanceOf(ToolResultBlock)
    const block = content as ToolResultBlock
    expect(block.content[0]).toBeInstanceOf(TextBlock)
    expect((block.content[0] as TextBlock).text).toContain('[Truncated:')
  })

  it('does not truncate results below threshold', async () => {
    const smallText = 'short result'
    const messages = [makeToolResultMessage(smallText)]
    const strategy = Offload.truncate('toolResults').when({ threshold: 2500 })
    const context = makeContext(messages)

    const result = await strategy.apply(context)

    expect(result).toBe(false)
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
    const strategy = Offload.truncate('toolResults')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('truncates error results when target is toolResultErrors', async () => {
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
    const strategy = Offload.truncate('toolResultErrors')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
  })

  it('skips already truncated results', async () => {
    const message = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'tool-123',
          status: 'success',
          content: [new TextBlock('[Truncated: 1 blocks, ~500 tokens]\npreview...')],
        }),
      ],
    })
    const strategy = Offload.truncate('toolResults')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('skips non-user messages when targeting tool results', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const message = new Message({
      role: 'assistant',
      content: [new TextBlock(largeText)],
    })
    const strategy = Offload.truncate('toolResults')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('truncates assistant text blocks with assistantMessages target', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const message = new Message({
      role: 'assistant',
      content: [new TextBlock(largeText)],
    })
    const strategy = Offload.truncate('assistantMessages')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = message.content[0] as TextBlock
    expect(block.text).toContain('[Truncated:')
  })

  it('truncates user text blocks with userMessages target', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const message = new Message({
      role: 'user',
      content: [new TextBlock(largeText)],
    })
    const strategy = Offload.truncate('userMessages')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = message.content[0] as TextBlock
    expect(block.text).toContain('[Truncated:')
  })

  it('does not truncate user messages with assistantMessages target', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const message = new Message({
      role: 'user',
      content: [new TextBlock(largeText)],
    })
    const strategy = Offload.truncate('assistantMessages')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('builds head-tail preview', async () => {
    const largeText = 'HEAD'.repeat(1000) + 'MIDDLE'.repeat(5000) + 'TAIL'.repeat(1000)
    const messages = [makeToolResultMessage(largeText)]
    const strategy = Offload.truncate('toolResults', { previewTokens: 100 })
    const context = makeContext(messages)

    await strategy.apply(context)

    const block = messages[0]!.content[0] as ToolResultBlock
    const previewText = (block.content[0] as TextBlock).text
    expect(previewText).toContain('[...')
    expect(previewText).toContain('chars elided')
  })

  it('returns false for empty messages', async () => {
    const strategy = Offload.truncate('toolResults')
    const context = makeContext([])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })
})

describe('Offload builder', () => {
  it('Offload() bare call creates a drop strategy', () => {
    const strategy = Offload('toolResults')
    expect(strategy.name).toBe('offload:drop')
    expect(strategy.init).toBeDefined()
    expect(strategy.apply).toBeDefined()
  })

  it('Offload() drops tool result content from L0 entirely', async () => {
    const largeText = 'x'.repeat(100)
    const messages = [makeToolResultMessage(largeText)]
    const strategy = Offload('toolResults')
    const context = makeContext(messages)

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = messages[0]!.content[0] as ToolResultBlock
    expect((block.content[0] as TextBlock).text).toBe('[Dropped]')
  })

  it('Offload() drops assistant text blocks', async () => {
    const message = new Message({
      role: 'assistant',
      content: [new TextBlock('some assistant response')],
    })
    const strategy = Offload('assistantMessages')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = message.content[0] as TextBlock
    expect(block.text).toBe('[Dropped]')
  })

  it('Offload() drops user text blocks', async () => {
    const message = new Message({
      role: 'user',
      content: [new TextBlock('some user message')],
    })
    const strategy = Offload('userMessages')
    const context = makeContext([message])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = message.content[0] as TextBlock
    expect(block.text).toBe('[Dropped]')
  })

  it('Offload.truncate with string[] target matches tool by name from assistant message', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const assistantMsg = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({ name: 'bash', toolUseId: 'tool-bash', input: {} }),
      ],
    })
    const userMsg = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'tool-bash',
          status: 'success',
          content: [new TextBlock(largeText)],
        }),
      ],
    })
    const strategy = Offload.truncate(['bash'])
    const context = makeContext([assistantMsg, userMsg])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    const block = userMsg.content[0] as ToolResultBlock
    expect((block.content[0] as TextBlock).text).toContain('[Truncated:')
  })

  it('Offload.truncate with string[] target skips non-matching tools', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const assistantMsg = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({ name: 'read_file', toolUseId: 'tool-read', input: {} }),
      ],
    })
    const userMsg = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'tool-read',
          status: 'success',
          content: [new TextBlock(largeText)],
        }),
      ],
    })
    const strategy = Offload.truncate(['bash'])
    const context = makeContext([assistantMsg, userMsg])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })

  it('Offload.summarize() creates a summarize strategy', () => {
    const strategy = Offload.summarize('toolResults')
    expect(strategy.name).toBe('offload:summarize')
    expect(strategy.apply).toBeDefined()
  })

  it('Offload.summarize().when() creates a strategy with conditions', () => {
    const strategy = Offload.summarize('toolResults').when({ utilization: 0.85 })
    expect(strategy.name).toBe('offload:summarize')
  })

  it('Offload.summarize(config) config-only creates untargeted strategy', () => {
    const strategy = Offload.summarize({ systemPrompt: 'summarize briefly' }).when({ utilization: 0.85 })
    expect(strategy.name).toBe('offload:summarize')
  })
})

describe('Offload with no target (fires on everything)', () => {
  it('Offload() with no target drops all content', async () => {
    const assistantMsg = new Message({
      role: 'assistant',
      content: [new TextBlock('assistant text')],
    })
    const userMsg = new Message({
      role: 'user',
      content: [
        new TextBlock('user text'),
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('tool output')],
        }),
      ],
    })
    const strategy = Offload()
    const context = makeContext([assistantMsg, userMsg])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    expect((assistantMsg.content[0] as TextBlock).text).toBe('[Dropped]')
    expect((userMsg.content[0] as TextBlock).text).toBe('[Dropped]')
    const toolBlock = userMsg.content[1] as ToolResultBlock
    expect((toolBlock.content[0] as TextBlock).text).toBe('[Dropped]')
  })

  it('Offload.truncate() with no target truncates all large content', async () => {
    const largeText = 'x'.repeat(2500 * 4 + 100)
    const assistantMsg = new Message({
      role: 'assistant',
      content: [new TextBlock(largeText)],
    })
    const userMsg = new Message({
      role: 'user',
      content: [
        new TextBlock(largeText),
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock(largeText)],
        }),
      ],
    })
    const strategy = Offload.truncate()
    const context = makeContext([assistantMsg, userMsg])

    const result = await strategy.apply(context)

    expect(result).toBe(true)
    expect((assistantMsg.content[0] as TextBlock).text).toContain('[Truncated:')
    expect((userMsg.content[0] as TextBlock).text).toContain('[Truncated:')
    const toolBlock = userMsg.content[1] as ToolResultBlock
    expect((toolBlock.content[0] as TextBlock).text).toContain('[Truncated:')
  })

  it('Offload.truncate() with no target and threshold skips small content', async () => {
    const smallText = 'short'
    const assistantMsg = new Message({
      role: 'assistant',
      content: [new TextBlock(smallText)],
    })
    const strategy = Offload.truncate().when({ threshold: 2500 })
    const context = makeContext([assistantMsg])

    const result = await strategy.apply(context)

    expect(result).toBe(false)
  })
})
