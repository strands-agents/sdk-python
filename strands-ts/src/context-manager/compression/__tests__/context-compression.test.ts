import { describe, it, expect, vi } from 'vitest'
import { ContextCompression } from '../context-compression.js'
import { Message, TextBlock } from '../../../types/messages.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../../../hooks/events.js'
import { ContextWindowOverflowError } from '../../../errors.js'
import { createMockAgent, invokeTrackedHook } from '../../../__fixtures__/agent-helpers.js'
import type { BaseModelConfig } from '../../../models/model.js'

function userMsg(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function assistantMsg(text: string): Message {
  return new Message({ role: 'assistant', content: [new TextBlock(text)] })
}

describe('ContextCompression', () => {
  describe('constructor', () => {
    it('validates proactive threshold must be > 0', () => {
      expect(() => new ContextCompression({ proactive: { threshold: 0 } })).toThrow(
        'proactive compression threshold must be between 0 (exclusive) and 1 (inclusive)'
      )
    })

    it('validates proactive threshold must be <= 1', () => {
      expect(() => new ContextCompression({ proactive: { threshold: 1.5 } })).toThrow(
        'proactive compression threshold must be between 0 (exclusive) and 1 (inclusive)'
      )
    })

    it('accepts valid proactive threshold', () => {
      expect(() => new ContextCompression({ proactive: { threshold: 0.8 } })).not.toThrow()
    })

    it('accepts threshold of exactly 1', () => {
      expect(() => new ContextCompression({ proactive: { threshold: 1 } })).not.toThrow()
    })

    it('defaults to truncate method', () => {
      const compression = new ContextCompression()
      expect((compression as any)._method).toBe('truncate')
    })

    it('accepts summarize method', () => {
      const compression = new ContextCompression({ method: 'summarize' })
      expect((compression as any)._method).toBe('summarize')
    })

    it('defaults proactive threshold to 0.7 when proactive is true or omitted', () => {
      const compression = new ContextCompression()
      expect((compression as any)._proactiveThreshold).toBe(0.7)
    })

    it('disables proactive compression when proactive is false', () => {
      const compression = new ContextCompression({ proactive: false })
      expect((compression as any)._proactiveThreshold).toBeUndefined()
    })
  })

  describe('initAgent', () => {
    it('registers AfterModelCallEvent hook', () => {
      const compression = new ContextCompression()
      const agent = createMockAgent()
      compression.initAgent(agent)

      const hookTypes = agent.trackedHooks.map((h) => h.eventType)
      expect(hookTypes).toContain(AfterModelCallEvent)
    })

    it('registers BeforeModelCallEvent hook', () => {
      const compression = new ContextCompression()
      const agent = createMockAgent()
      compression.initAgent(agent)

      const hookTypes = agent.trackedHooks.map((h) => h.eventType)
      expect(hookTypes).toContain(BeforeModelCallEvent)
    })

    it('registers AfterInvocationEvent hook when method is truncate', () => {
      const compression = new ContextCompression({ method: 'truncate' })
      const agent = createMockAgent()
      compression.initAgent(agent)

      const hookTypes = agent.trackedHooks.map((h) => h.eventType)
      expect(hookTypes).toContain(AfterInvocationEvent)
    })

    it('does not register AfterInvocationEvent hook when method is summarize', () => {
      const compression = new ContextCompression({ method: 'summarize' })
      const agent = createMockAgent()
      compression.initAgent(agent)

      const hookTypes = agent.trackedHooks.map((h) => h.eventType)
      expect(hookTypes).not.toContain(AfterInvocationEvent)
    })
  })

  describe('proactive hook (BeforeModelCallEvent)', () => {
    it('skips when proactive threshold is undefined', async () => {
      const compression = new ContextCompression({ proactive: false })
      const messages = Array.from({ length: 50 }, (_, i) =>
        i % 2 === 0 ? userMsg(`msg ${i}`) : assistantMsg(`resp ${i}`)
      )
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any
      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 900, // 90% of limit
      })
      await invokeTrackedHook(mockAgent, event)

      // Messages should not have been modified
      expect(mockAgent.messages).toHaveLength(50)
    })

    it('triggers reduce when ratio exceeds threshold', async () => {
      const compression = new ContextCompression({
        proactive: { threshold: 0.7 },
        windowSize: 4,
      })
      const messages = [
        userMsg('Message 1'),
        assistantMsg('Response 1'),
        userMsg('Message 2'),
        assistantMsg('Response 2'),
        userMsg('Message 3'),
        assistantMsg('Response 3'),
      ]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any
      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 800, // 80% > 70% threshold
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages.length).toBeLessThan(6)
    })

    it('does not trigger reduce when ratio is below threshold', async () => {
      const compression = new ContextCompression({ proactive: { threshold: 0.7 } })
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any
      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 500, // 50% < 70% threshold
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages).toHaveLength(2)
    })

    it('uses estimateInputTokens when projectedInputTokens is not provided', async () => {
      const compression = new ContextCompression({
        proactive: { threshold: 0.7 },
        windowSize: 2,
      })
      const messages = [
        userMsg('Message 1'),
        new Message({
          role: 'assistant',
          content: [new TextBlock('Response 1')],
          metadata: { usage: { inputTokens: 600, outputTokens: 200, totalTokens: 800 } },
        }),
        userMsg('Message 2'),
        assistantMsg('Response 2'),
        userMsg('Message 3'),
      ]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const mockModel = {
        getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig,
        countTokens: vi.fn().mockResolvedValue(100),
      } as any
      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        // No projectedInputTokens — will use estimateInputTokens
      })
      await invokeTrackedHook(mockAgent, event)

      // 600+200+countTokens(remaining) = 900 > 700 threshold — should compress
      expect(mockAgent.messages.length).toBeLessThan(5)
    })

    it('skips when projectedInputTokens is undefined and estimation returns undefined', async () => {
      const compression = new ContextCompression({ proactive: { threshold: 0.7 } })
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const mockModel = {
        getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig,
        countTokens: vi.fn().mockRejectedValue(new Error('fail')),
      } as any
      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages).toHaveLength(2)
    })
  })

  describe('reactive hook (AfterModelCallEvent)', () => {
    it('retries on ContextWindowOverflowError', async () => {
      const compression = new ContextCompression({ windowSize: 2 })
      const messages = [
        userMsg('Message 1'),
        assistantMsg('Response 1'),
        userMsg('Message 2'),
        assistantMsg('Response 2'),
      ]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterModelCallEvent({
        agent: mockAgent,
        model: {} as any,
        attemptCount: 1,
        error: new ContextWindowOverflowError('overflow'),
        invocationState: {},
      })
      await invokeTrackedHook(mockAgent, event)

      expect(event.retry).toBe(true)
      expect(mockAgent.messages.length).toBeLessThan(4)
    })

    it('does not set retry when error is not ContextWindowOverflowError', async () => {
      const compression = new ContextCompression()
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterModelCallEvent({
        agent: mockAgent,
        model: {} as any,
        attemptCount: 1,
        error: new Error('some other error'),
        invocationState: {},
      })
      await invokeTrackedHook(mockAgent, event)

      expect(event.retry).toBeUndefined()
    })

    it('does not set retry when no error is present', async () => {
      const compression = new ContextCompression()
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterModelCallEvent({
        agent: mockAgent,
        model: {} as any,
        attemptCount: 1,
        invocationState: {},
      })
      await invokeTrackedHook(mockAgent, event)

      expect(event.retry).toBeUndefined()
    })

    it('does not set retry when reduce returns false', async () => {
      const compression = new ContextCompression({ windowSize: 10 })
      // Only 2 messages - truncate returns false for messages.length <= 2
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterModelCallEvent({
        agent: mockAgent,
        model: {} as any,
        attemptCount: 1,
        error: new ContextWindowOverflowError('overflow'),
        invocationState: {},
      })
      await invokeTrackedHook(mockAgent, event)

      expect(event.retry).toBeUndefined()
    })
  })

  describe('AfterInvocationEvent hook (sliding window enforcement)', () => {
    it('truncates when messages exceed window size', async () => {
      const compression = new ContextCompression({ windowSize: 4 })
      const messages = [
        userMsg('Message 1'),
        assistantMsg('Response 1'),
        userMsg('Message 2'),
        assistantMsg('Response 2'),
        userMsg('Message 3'),
        assistantMsg('Response 3'),
      ]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterInvocationEvent({
        agent: mockAgent,
        invocationState: {},
      })

      // Find AfterInvocationEvent hook specifically
      const hook = mockAgent.trackedHooks.find((h) => h.eventType === AfterInvocationEvent)
      expect(hook).toBeDefined()
      await hook!.callback(event)

      expect(mockAgent.messages.length).toBeLessThanOrEqual(4)
    })

    it('does not truncate when messages are within window size', async () => {
      const compression = new ContextCompression({ windowSize: 10 })
      const messages = [userMsg('Message 1'), assistantMsg('Response 1')]
      const mockAgent = createMockAgent({ messages })
      compression.initAgent(mockAgent)

      const event = new AfterInvocationEvent({
        agent: mockAgent,
        invocationState: {},
      })

      const hook = mockAgent.trackedHooks.find((h) => h.eventType === AfterInvocationEvent)
      expect(hook).toBeDefined()
      await hook!.callback(event)

      expect(mockAgent.messages).toHaveLength(2)
    })
  })

  describe('getTools', () => {
    it('returns empty array', () => {
      const compression = new ContextCompression()
      expect(compression.getTools()).toEqual([])
    })
  })
})
