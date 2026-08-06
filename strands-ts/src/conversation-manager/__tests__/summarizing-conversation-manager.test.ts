import { describe, it, expect, vi } from 'vitest'
import { SummarizingConversationManager } from '../summarizing-conversation-manager.js'
import { ContextWindowOverflowError, Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../index.js'
import { AfterModelCallEvent, BeforeModelCallEvent } from '../../hooks/events.js'
import { createMockAgent, invokeTrackedHook } from '../../__fixtures__/agent-helpers.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { DEFAULT_SUMMARIZATION_INSTRUCTION, DEFAULT_SUMMARIZATION_PROMPT } from '../compression/context-compression.js'
import { logger } from '../../logging/logger.js'
import type { Model, BaseModelConfig, StreamOptions } from '../../models/model.js'
import type { ToolSpec } from '../../tools/types.js'
import type { ToolRegistry } from '../../registry/tool-registry.js'

function textMsg(role: 'user' | 'assistant', text: string): Message {
  return new Message({ role, content: [new TextBlock(text)] })
}

function makeMessages(count: number): Message[] {
  return Array.from({ length: count }, (_, i) => textMsg(i % 2 === 0 ? 'user' : 'assistant', `Message ${i + 1}`))
}

describe('SummarizingConversationManager', () => {
  describe('constructor', () => {
    it('clamps summaryRatio to [0.1, 0.8]', () => {
      expect((new SummarizingConversationManager({ summaryRatio: 0 }) as any)._summaryRatio).toBe(0.1)
      expect((new SummarizingConversationManager({ summaryRatio: 1.0 }) as any)._summaryRatio).toBe(0.8)
    })
  })

  describe('reduce', () => {
    it('summarizes oldest messages and replaces them with a user-role summary', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary of conversation' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const messages = makeMessages(20)
      const lastTwo = messages.slice(-2)
      const mockAgent = createMockAgent({ messages })

      const result = await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      // 20 * 0.5 = 10 summarized → 1 summary + 10 remaining = 11
      expect(mockAgent.messages).toHaveLength(11)
      expect(mockAgent.messages[0]!.role).toBe('user')
      expect(mockAgent.messages[0]!.content[0]!).toEqual({
        type: 'textBlock',
        text: 'Summary of conversation',
      })
      // Recent messages preserved
      expect(mockAgent.messages.slice(-2)).toEqual(lastTwo)
    })

    it('assigns a durable tracking id to the generated summary message', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary of conversation' })

      const manager = new SummarizingConversationManager({ summaryRatio: 0.5, preserveRecentMessages: 2 })
      const mockAgent = createMockAgent({ messages: makeMessages(20) })

      await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(typeof mockAgent.messages[0]!.trackingId).toBe('string')
      expect(mockAgent.messages[0]!.trackingId).toBeTruthy()
    })

    it('uses the config model over the reduce model when provided', async () => {
      const configModel = new MockMessageModel()
      configModel.addTurn({ type: 'textBlock', text: 'Config model summary' })
      const reduceModel = new MockMessageModel()
      reduceModel.addTurn({ type: 'textBlock', text: 'Reduce model summary' })

      const manager = new SummarizingConversationManager({
        model: configModel as unknown as Model,
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })

      await manager.reduce({
        agent: mockAgent,
        model: reduceModel as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(mockAgent.messages[0]!.content[0]!).toEqual({
        type: 'textBlock',
        text: 'Config model summary',
      })
    })

    it('uses the config model when no reduce model is provided', async () => {
      const configModel = new MockMessageModel()
      configModel.addTurn({ type: 'textBlock', text: 'Config model summary' })

      const manager = new SummarizingConversationManager({
        model: configModel as unknown as Model,
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })

      const result = await manager.reduce({
        agent: mockAgent,
        model: {} as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({
        type: 'textBlock',
        text: 'Config model summary',
      })
    })

    it('returns false when there are not enough messages to summarize', async () => {
      const model = new MockMessageModel()
      const manager = new SummarizingConversationManager({
        preserveRecentMessages: 10,
      })
      const messages = makeMessages(8)
      const mockAgent = createMockAgent({ messages })

      const result = await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(false)
      expect(mockAgent.messages).toHaveLength(8)
    })

    it('rethrows model errors with the overflow error as cause', async () => {
      const model = new MockMessageModel()
      model.addTurn(new Error('model failed'))

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const overflowError = new ContextWindowOverflowError('overflow')
      const mockAgent = createMockAgent({ messages: makeMessages(20) })

      const thrown = await manager
        .reduce({ agent: mockAgent, model: model as unknown as Model, error: overflowError })
        .catch((e: unknown) => e)
      expect(thrown).toBeInstanceOf(Error)
      expect((thrown as Error).message).toBe('model failed')
      expect((thrown as Error).cause).toBe(overflowError)
    })

    it('wraps non-Error throw values with the overflow error as cause', async () => {
      const model = new MockMessageModel()
      const err = 'string error'
      vi.spyOn(model, 'streamAggregated').mockImplementation(async function* () {
        yield undefined as any
        throw err
      } as any)

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const overflowError = new ContextWindowOverflowError('overflow')
      const mockAgent = createMockAgent({ messages: makeMessages(20) })

      const thrown = await manager
        .reduce({ agent: mockAgent, model: model as unknown as Model, error: overflowError })
        .catch((e: unknown) => e)
      expect(thrown).toBeInstanceOf(Error)
      expect((thrown as Error).message).toBe('string error')
      expect((thrown as Error).cause).toBe(overflowError)
    })

    it('passes the correct message slice and system prompt to the model', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'stream')

      const customPrompt = 'Custom summarization prompt'
      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        summarizationSystemPrompt: customPrompt,
      })
      const messages = makeMessages(10)
      const expectedSlice = messages.slice(0, 5)
      const mockAgent = createMockAgent({ messages })

      await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(streamSpy).toHaveBeenCalledOnce()
      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]!
      // First 5 messages (10 * 0.5) plus the "Please summarize" request
      expect(calledMessages).toHaveLength(6)
      expect(calledMessages!.slice(0, 5)).toEqual(expectedSlice)
      expect(calledMessages![5]!.role).toBe('user')
      expect(calledMessages![5]!.content[0]!).toEqual(
        expect.objectContaining({ text: 'Please summarize this conversation.' })
      )
      expect(calledOptions).toEqual(expect.objectContaining({ systemPrompt: customPrompt }))
    })

    it('preserveRecentMessages dominates when larger than ratio allows', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.8,
        preserveRecentMessages: 18,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })

      const result = await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      // 20 * 0.8 = 16, but min(16, 20-18) = 2, so only 2 summarized
      // 1 summary + 18 remaining = 19
      expect(mockAgent.messages).toHaveLength(19)
    })
  })

  describe('tool pair adjustment', () => {
    it('advances split point past orphaned toolResult and toolUse boundaries', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.3,
        preserveRecentMessages: 2,
      })

      // Natural split at ~index 3 lands on a toolResult
      const messages = [
        textMsg('user', 'Message 1'),
        textMsg('assistant', 'Message 2'),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [new ToolResultBlock({ toolUseId: 'id-1', status: 'success', content: [new TextBlock('Result')] })],
        }),
        textMsg('assistant', 'Response after tool'),
        ...makeMessages(8),
      ]
      const mockAgent = createMockAgent({ messages })

      const result = await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      // After summary insertion, no remaining message should start with an orphaned toolResult
      expect(mockAgent.messages[1]!.content.some((b) => b.type === 'toolResultBlock')).toBe(false)
    })

    it('throws when no valid split point exists', async () => {
      const model = new MockMessageModel()
      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 0,
      })

      // All messages are toolResults
      const messages = Array.from(
        { length: 4 },
        (_, i) =>
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({ toolUseId: `id-${i}`, status: 'success', content: [new TextBlock(`R${i}`)] }),
            ],
          })
      )
      const mockAgent = createMockAgent({ messages })

      await expect(
        manager.reduce({
          agent: mockAgent,
          model: model as unknown as Model,
          error: new ContextWindowOverflowError('overflow'),
        })
      ).rejects.toThrow('Unable to find valid split point for summarization')
    })
  })

  describe('base class hook integration', () => {
    // Two agents: pluginAgent receives the hook registration via initAgent(),
    // while agent holds the messages and is carried on the event object.
    it('async reduce sets retry=true through the base class await', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })
      const messages = makeMessages(20)
      const agent = createMockAgent({ messages })

      const pluginAgent = createMockAgent()
      manager.initAgent(pluginAgent)
      const event = new AfterModelCallEvent({
        agent,
        model: model as unknown as Model,
        attemptCount: 1,
        error: new ContextWindowOverflowError('overflow'),
        invocationState: {},
      })
      await invokeTrackedHook(pluginAgent, event)

      expect(event.retry).toBe(true)
      expect(agent.messages).toHaveLength(11)
    })
  })

  describe('reduceOnThreshold', () => {
    it('summarizes oldest messages when compressionThreshold is exceeded', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary of conversation' })

      const manager = new SummarizingConversationManager({
        model: model as unknown as Model,
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: { compressionThreshold: 0.7 },
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })
      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any

      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 800,
      })
      await invokeTrackedHook(mockAgent, event)

      // 20 * 0.5 = 10 summarized → 1 summary + 10 remaining = 11
      expect(mockAgent.messages).toHaveLength(11)
      expect(mockAgent.messages[0]!.role).toBe('user')
      expect(mockAgent.messages[0]!.content[0]!).toEqual({
        type: 'textBlock',
        text: 'Summary of conversation',
      })
    })

    it('does not summarize when below compressionThreshold', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        model: model as unknown as Model,
        proactiveCompression: { compressionThreshold: 0.7 },
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })
      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any

      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 500,
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages).toHaveLength(20)
    })

    it('returns false and does not throw when summarization fails', async () => {
      const model = new MockMessageModel()
      model.addTurn(new Error('model failed'))

      const manager = new SummarizingConversationManager({
        model: model as unknown as Model,
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: { compressionThreshold: 0.7 },
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({ messages })
      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any

      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 800,
      })

      // Should not throw — reduceOnThreshold is best-effort
      await invokeTrackedHook(mockAgent, event)
      expect(mockAgent.messages).toHaveLength(20)
    })
  })

  describe('pinFirst', () => {
    it('protects first N messages from summarization', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        pinFirst: 2,
      })

      const agent = createMockAgent({
        messages: [
          textMsg('user', 'protected-1'),
          textMsg('assistant', 'protected-2'),
          textMsg('user', 'summarize-me'),
          textMsg('assistant', 'summarize-me-too'),
          textMsg('user', 'recent-1'),
          textMsg('assistant', 'recent-2'),
        ],
      })

      await manager.reduce({ agent, model: model as unknown as Model })

      const texts = agent.messages.map((m) => (m.content[0] as TextBlock).text)
      expect(texts).toEqual(['protected-1', 'protected-2', 'Summary', 'summarize-me-too', 'recent-1', 'recent-2'])
    })

    it('returns false when all messages in summary range are protected', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.3,
        preserveRecentMessages: 2,
        pinFirst: 10,
      })

      const agent = createMockAgent({ messages: makeMessages(6) })
      const result = await manager.reduce({ agent, model: model as unknown as Model })
      expect(result).toBe(false)
    })

    it('pinned message in middle survives summarization', async () => {
      const { pinMessage } = await import('../compression/pin-message.js')
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
      })

      const messages = [
        textMsg('user', 'old-1'),
        textMsg('assistant', 'pinned-middle'),
        textMsg('user', 'old-3'),
        textMsg('assistant', 'old-4'),
        textMsg('user', 'recent-1'),
        textMsg('assistant', 'recent-2'),
      ]
      pinMessage(messages, 1)
      const agent = createMockAgent({ messages })

      await manager.reduce({ agent, model: model as unknown as Model })

      const texts = agent.messages.map((m) => (m.content[0] as TextBlock).text)
      expect(texts).toEqual(['pinned-middle', 'Summary', 'old-4', 'recent-1', 'recent-2'])
    })
  })

  describe('cacheAligned', () => {
    const toolSpecs: ToolSpec[] = [{ name: 'search', description: 'search tool', inputSchema: { type: 'object' } }]

    function stubRegistry(specs: ToolSpec[]): ToolRegistry {
      return {
        list: () => specs.map((toolSpec) => ({ toolSpec })),
      } as unknown as ToolRegistry
    }

    it('proactive reduce sends the live systemPrompt, toolSpecs, and full history with a trailing instruction', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary of conversation' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      const messages = makeMessages(20)
      const lastTwo = messages.slice(-2)
      const fullHistory = [...messages]
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      // Proactive: no error passed
      const result = await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      expect(result).toBe(true)
      expect(streamSpy).toHaveBeenCalledOnce()
      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      // Full history (20) plus the trailing instruction turn
      expect(calledMessages).toHaveLength(21)
      expect(calledMessages.slice(0, 20)).toEqual(fullHistory)
      expect(calledMessages[20]!.role).toBe('user')
      expect(calledOptions.systemPrompt).toBe('Live system prompt')
      expect(calledOptions.toolSpecs).toEqual(toolSpecs)

      // 20 * 0.5 = 10 summarized → 1 summary + 10 remaining = 11; recent messages preserved verbatim
      expect(mockAgent.messages).toHaveLength(11)
      expect(mockAgent.messages[0]!.role).toBe('user')
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Summary of conversation' })
      expect(mockAgent.messages.slice(-2)).toEqual(lastTwo)
    })

    it('proactive aligned request carries DEFAULT_SUMMARIZATION_INSTRUCTION as the trailing turn by default', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      // Assistant tail: the instruction is appended as its own user turn
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      const [calledMessages] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      const trailing = calledMessages[calledMessages.length - 1]!
      expect(trailing.role).toBe('user')
      expect(trailing.content).toHaveLength(1)
      // The purpose-built user-turn instruction, not the summarization system prompt
      expect((trailing.content[0] as TextBlock).text).toBe(DEFAULT_SUMMARIZATION_INSTRUCTION)
    })

    it('proactive aligned request carries a user-supplied summarizationSystemPrompt as the instruction', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
        summarizationSystemPrompt: 'Custom summarization override',
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      const [calledMessages] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      const trailing = calledMessages[calledMessages.length - 1]!
      expect(trailing.role).toBe('user')
      expect((trailing.content[0] as TextBlock).text).toBe('Custom summarization override')
    })

    it('proactive trigger through the BeforeModelCallEvent hook merges the instruction into the user-message tail', async () => {
      const model = new MockMessageModel()
      model.updateConfig({ modelId: 'test-model', contextWindowLimit: 1000 })
      model.addTurn({ type: 'textBlock', text: 'Summary of conversation' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: { compressionThreshold: 0.7 },
        cacheAligned: true,
      })
      // 21 messages: at the proactive trigger the tail is always the upcoming turn's USER message
      const messages = makeMessages(21)
      const liveHistory = [...messages]
      expect(liveHistory[20]!.role).toBe('user')
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })
      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: model as unknown as Model,
        invocationState: {},
        projectedInputTokens: 800,
      })
      await invokeTrackedHook(mockAgent, event)

      expect(streamSpy).toHaveBeenCalledOnce()
      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]

      // (a) No consecutive user turns: roles strictly alternate across the built request
      expect(calledMessages).toHaveLength(21)
      for (let i = 1; i < calledMessages.length; i++) {
        expect(calledMessages[i]!.role).not.toBe(calledMessages[i - 1]!.role)
      }

      // (b) The cached prefix is byte-identical: same message instances as the live history
      for (let i = 0; i < 20; i++) {
        expect(calledMessages[i]).toBe(liveHistory[i])
      }

      // (c) The instruction rides in the final user message, after the original content
      const finalMessage = calledMessages[20]!
      expect(finalMessage.role).toBe('user')
      expect(finalMessage.content).toHaveLength(2)
      expect((finalMessage.content[0] as TextBlock).text).toBe('Message 21')
      expect((finalMessage.content[1] as TextBlock).text).toBe(DEFAULT_SUMMARIZATION_INSTRUCTION)
      // The live history's tail message was cloned, not mutated
      expect(finalMessage).not.toBe(liveHistory[20])
      expect(liveHistory[20]!.content).toHaveLength(1)

      // Live prefix reused as-is
      expect(calledOptions.systemPrompt).toBe('Live system prompt')
      expect(calledOptions.toolSpecs).toEqual(toolSpecs)

      // 21 * 0.5 = 10 summarized → 1 summary + 11 remaining = 12
      expect(mockAgent.messages).toHaveLength(12)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Summary of conversation' })
    })

    it('reactive reduce uses the slice-based path even when cacheAligned is set', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      const messages = makeMessages(20)
      const expectedSlice = messages.slice(0, 10)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      // Reactive: error passed
      await manager.reduce({
        agent: mockAgent,
        model: model as unknown as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(streamSpy).toHaveBeenCalledOnce()
      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      // Slice-based: only the summarize slice + the "Please summarize" request, live tool specs NOT sent
      expect(calledMessages).toHaveLength(11)
      expect(calledMessages.slice(0, 10)).toEqual(expectedSlice)
      expect((calledMessages[10]!.content[0] as TextBlock).text).toBe('Please summarize this conversation.')
      expect(calledOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(calledOptions.toolSpecs).toBeUndefined()
    })

    it('proactive reduce with cacheAligned off is identical to the slice-based path', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
      })
      const messages = makeMessages(20)
      const expectedSlice = messages.slice(0, 10)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      expect(calledMessages).toHaveLength(11)
      expect(calledMessages.slice(0, 10)).toEqual(expectedSlice)
      expect(calledOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(calledOptions.toolSpecs).toBeUndefined()
      expect(mockAgent.messages).toHaveLength(11)
    })

    it('falls back to slice-based summarization when the aligned response contains tool use', async () => {
      const model = new MockMessageModel()
      // Aligned request carries live tool specs, so the model may answer with a tool call
      model.addTurn({ type: 'toolUseBlock', name: 'search', toolUseId: 'id-1', input: {} })
      model.addTurn({ type: 'textBlock', text: 'Slice summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      const result = await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      expect(result).toBe(true)
      expect(streamSpy).toHaveBeenCalledTimes(2)
      // First call: aligned (full history + trailing instruction, live tool specs)
      const [alignedMessages, alignedOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      expect(alignedMessages).toHaveLength(21)
      expect(alignedOptions.toolSpecs).toEqual(toolSpecs)
      // Second call: slice-based fallback
      const [sliceMessages, sliceOptions] = streamSpy.mock.calls[1]! as [Message[], StreamOptions]
      expect(sliceMessages).toHaveLength(11)
      expect(sliceOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(sliceOptions.toolSpecs).toBeUndefined()
      // The slice summary landed; no dangling toolUse was spliced into history
      expect(mockAgent.messages).toHaveLength(11)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Slice summary' })
      expect(mockAgent.messages.every((m) => !m.content.some((b) => b.type === 'toolUseBlock'))).toBe(true)
    })

    it('falls back to slice-based summarization when the aligned request fails', async () => {
      const model = new MockMessageModel()
      model.addTurn(new Error('aligned request failed'))
      model.addTurn({ type: 'textBlock', text: 'Slice summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      const result = await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      // Compaction still happens via the slice-based path
      expect(result).toBe(true)
      expect(streamSpy).toHaveBeenCalledTimes(2)
      const [sliceMessages, sliceOptions] = streamSpy.mock.calls[1]! as [Message[], StreamOptions]
      expect(sliceMessages).toHaveLength(11)
      expect(sliceOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(mockAgent.messages).toHaveLength(11)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Slice summary' })
    })

    it('uses the slice-based path when a model override is configured, even with cacheAligned set', async () => {
      const configModel = new MockMessageModel()
      configModel.addTurn({ type: 'textBlock', text: 'Override summary' })
      const streamSpy = vi.spyOn(configModel, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        model: configModel as unknown as Model,
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      const messages = makeMessages(20)
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      const result = await manager.reduce({ agent: mockAgent, model: {} as Model })

      expect(result).toBe(true)
      expect(streamSpy).toHaveBeenCalledOnce()
      // Slice-based: sending the full live history to a different model guarantees a cache miss
      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      expect(calledMessages).toHaveLength(11)
      expect(calledOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(calledOptions.toolSpecs).toBeUndefined()
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Override summary' })
    })

    it('warns at construction when cacheAligned is combined with a model override', () => {
      const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      try {
        new SummarizingConversationManager({
          model: new MockMessageModel() as unknown as Model,
          proactiveCompression: true,
          cacheAligned: true,
        })
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('model override defeats cache alignment'))
      } finally {
        warnSpy.mockRestore()
      }
    })

    it('warns at construction when cacheAligned is set without proactive compression', () => {
      const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      try {
        new SummarizingConversationManager({ cacheAligned: true })
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('only takes effect on proactive compression'))
      } finally {
        warnSpy.mockRestore()
      }
    })

    it('proactive reduce with a dangling tool-use tail falls back to the slice-based path', async () => {
      const model = new MockMessageModel()
      model.addTurn({ type: 'textBlock', text: 'Summary' })
      const streamSpy = vi.spyOn(model, 'streamAggregated')

      const manager = new SummarizingConversationManager({
        summaryRatio: 0.5,
        preserveRecentMessages: 2,
        proactiveCompression: true,
        cacheAligned: true,
      })
      // Last message is an assistant toolUse still awaiting its result — not a valid append tail
      const messages = [
        ...makeMessages(19),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'search', toolUseId: 'id-1', input: {} })],
        }),
      ]
      const originalLength = messages.length
      const mockAgent = createMockAgent({
        messages,
        extra: { systemPrompt: 'Live system prompt', toolRegistry: stubRegistry(toolSpecs) },
      })

      await manager.reduce({ agent: mockAgent, model: model as unknown as Model })

      const [calledMessages, calledOptions] = streamSpy.mock.calls[0]! as [Message[], StreamOptions]
      // Slice-based fallback: does not send the full history / live tool specs
      expect(calledMessages.length).toBeLessThan(originalLength)
      expect(calledOptions.systemPrompt).toBe(DEFAULT_SUMMARIZATION_PROMPT)
      expect(calledOptions.toolSpecs).toBeUndefined()
    })
  })
})
