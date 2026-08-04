import { describe, it, expect, vi } from 'vitest'
import { createTokenUsageMiddleware } from '../../context-manager/modes/agentic/agentic-context.js'
import { Message, TextBlock } from '../../types/messages.js'
import type { InvokeModelContext } from '../../middleware/stages.js'
import type { Model } from '../../models/model.js'
import { Agent } from '../../agent/agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { ContextInjector } from '../../vended-plugins/context-injector/plugin.js'

function createMockModel(contextWindowLimit?: number): Model {
  return {
    getConfig: () => ({ contextWindowLimit }),
  } as unknown as Model
}

function createContext(overrides: Partial<InvokeModelContext> = {}): InvokeModelContext {
  return {
    agent: {} as InvokeModelContext['agent'],
    messages: [new Message({ role: 'user', content: [new TextBlock('hello')] })],
    toolSpecs: [],
    invocationState: {},
    ...overrides,
  }
}

describe('createTokenUsageMiddleware', () => {
  it('returns context unchanged when projectedInputTokens is not set', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(200_000))
    const context = createContext()

    const result = await middleware(context)

    expect(result).toBe(context)
  })

  it('returns context unchanged when messages are empty', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(200_000))
    const context = createContext({
      messages: [],
      projectedInputTokens: 50_000,
    })

    const result = await middleware(context)

    expect(result).toBe(context)
  })

  it('appends context status to last message content', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(200_000))
    const context = createContext({
      projectedInputTokens: 50_000,
    })

    const result = await middleware(context)

    expect(result.messages.length).toBe(1)
    const lastMsg = result.messages[0]!
    // The status text carries a live token count that changes every call, so it is appended behind a
    // cachePoint boundary to keep it out of the prompt cache's reusable prefix.
    expect(lastMsg.content.map((block) => block.type)).toStrictEqual(['textBlock', 'cachePointBlock', 'textBlock'])
    const statusBlock = lastMsg.content[lastMsg.content.length - 1] as TextBlock
    expect(statusBlock.text).toContain('<context-status>')
    expect(statusBlock.text).toContain('25.0%')
    expect(statusBlock.text).toContain('<remaining>')
    expect(statusBlock.text).toContain('</context-status>')
  })

  it('does not mutate the original messages array', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(200_000))
    const originalMessage = new Message({ role: 'user', content: [new TextBlock('hello')] })
    const context = createContext({
      messages: [originalMessage],
      projectedInputTokens: 50_000,
    })

    const result = await middleware(context)

    expect(result.messages).not.toBe(context.messages)
    expect(originalMessage.content.length).toBe(1)
  })

  it('uses DEFAULT_CONTEXT_WINDOW_LIMIT when model config has no limit', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(undefined))
    const context = createContext({
      projectedInputTokens: 100_000,
    })

    const result = await middleware(context)

    const statusBlock = (result.messages[0]! as Message).content[
      (result.messages[0]! as Message).content.length - 1
    ] as TextBlock
    expect(statusBlock.text).toContain('<context-status>')
    expect(statusBlock.text).toContain('50.0%')
    expect(statusBlock.text).toContain('200,000')
  })

  it('preserves message metadata', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(200_000))
    const metadata = { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } }
    const context = createContext({
      messages: [new Message({ role: 'user', content: [new TextBlock('hello')], metadata })],
      projectedInputTokens: 50_000,
    })

    const result = await middleware(context)

    expect(result.messages[0]!.metadata).toEqual(metadata)
  })

  it('reports correct remaining tokens', async () => {
    const middleware = createTokenUsageMiddleware(createMockModel(100_000))
    const context = createContext({
      projectedInputTokens: 80_000,
    })

    const result = await middleware(context)

    const statusBlock = (result.messages[0]! as Message).content[
      (result.messages[0]! as Message).content.length - 1
    ] as TextBlock
    expect(statusBlock.text).toContain('80.0%')
    expect(statusBlock.text).toContain('<remaining>~20,000 tokens</remaining>')
  })
})

describe('token usage middleware integration', () => {
  it('injects context status into messages sent to model on second call', async () => {
    const model = new MockMessageModel()
      .addTurn(
        { type: 'textBlock', text: 'First response' },
        { usage: { inputTokens: 1000, outputTokens: 200, totalTokens: 1200 } }
      )
      .addTurn(
        { type: 'textBlock', text: 'Second response' },
        { usage: { inputTokens: 2000, outputTokens: 300, totalTokens: 2300 } }
      )

    model.updateConfig({ contextWindowLimit: 100_000 })

    const agent = new Agent({ model, contextManager: 'agentic', printer: false })
    const streamSpy = vi.spyOn(model, 'stream')

    await agent.invoke('First message')
    await agent.invoke('Second message')

    const secondCallMessages = streamSpy.mock.calls[1]![0] as Message[]
    const lastMessage = secondCallMessages[secondCallMessages.length - 1]!
    const lastBlock = lastMessage.content[lastMessage.content.length - 1] as TextBlock
    expect(lastBlock.text).toContain('<context-status>')
    expect(lastBlock.text).toContain('100,000')
    expect(lastBlock.text).toContain('<remaining>')
  })

  it('injects on first call when token estimation succeeds', async () => {
    const model = new MockMessageModel().addTurn(
      { type: 'textBlock', text: 'First response' },
      { usage: { inputTokens: 1000, outputTokens: 200, totalTokens: 1200 } }
    )

    model.updateConfig({ contextWindowLimit: 100_000 })

    const agent = new Agent({ model, contextManager: 'agentic', printer: false })
    const streamSpy = vi.spyOn(model, 'stream')

    await agent.invoke('First message')

    const firstCallMessages = streamSpy.mock.calls[0]![0] as Message[]
    const lastMessage = firstCallMessages[firstCallMessages.length - 1]!
    const lastBlock = lastMessage.content[lastMessage.content.length - 1] as TextBlock
    expect(lastBlock.text).toContain('<context-status>')
    expect(lastBlock.text).toContain('100,000')
  })

  it('does not register middleware when contextManager is not agentic', async () => {
    const model = new MockMessageModel()
      .addTurn(
        { type: 'textBlock', text: 'First response' },
        { usage: { inputTokens: 1000, outputTokens: 200, totalTokens: 1200 } }
      )
      .addTurn(
        { type: 'textBlock', text: 'Second response' },
        { usage: { inputTokens: 2000, outputTokens: 300, totalTokens: 2300 } }
      )

    model.updateConfig({ contextWindowLimit: 100_000 })

    const agent = new Agent({ model, printer: false })
    const streamSpy = vi.spyOn(model, 'stream')

    await agent.invoke('First message')
    await agent.invoke('Second message')

    const secondCallMessages = streamSpy.mock.calls[1]![0] as Message[]
    const lastMessage = secondCallMessages[secondCallMessages.length - 1]!
    const lastBlock = lastMessage.content[lastMessage.content.length - 1] as TextBlock
    expect(lastBlock.text).not.toContain('<context-status>')
  })

  it('shares one cache boundary with an injector so both volatile blocks stay outside the prefix', async () => {
    const model = new MockMessageModel().addTurn(
      { type: 'textBlock', text: 'response' },
      { usage: { inputTokens: 1000, outputTokens: 100, totalTokens: 1100 } }
    )
    model.updateConfig({ contextWindowLimit: 100_000 })

    const agent = new Agent({
      model,
      contextManager: 'agentic',
      plugins: [new ContextInjector({ renderContent: async () => 'INJECTED' })],
      printer: false,
    })
    const streamSpy = vi.spyOn(model, 'stream')

    await agent.invoke('ask')

    const sent = streamSpy.mock.calls[0]![0] as Message[]
    const lastMessage = sent[sent.length - 1]!
    expect(lastMessage.content.map((block) => block.type)).toStrictEqual([
      'textBlock', // the durable ask
      'cachePointBlock', // the single boundary
      'textBlock', // <context-status>
      'textBlock', // injected text
    ])
  })
})
