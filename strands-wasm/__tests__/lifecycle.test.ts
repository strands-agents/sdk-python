import { describe, it, expect } from 'vitest'
import { LifecycleBridge } from '../entry'
import { Agent, FunctionTool } from '@strands-agents/sdk'
import { MockMessageModel } from '$/fixtures/mock-message-model'

describe('LifecycleBridge', () => {
  async function runTextTurn(): Promise<LifecycleBridge> {
    const bridge = new LifecycleBridge()
    const model = new MockMessageModel()
    model.addTurn({ type: 'textBlock', text: 'Hello' })
    const agent = new Agent({ model, plugins: [bridge], printer: false })
    await agent.invoke('hello')
    return bridge
  }

  describe('Plugin interface', () => {
    it('has name property', () => {
      const bridge = new LifecycleBridge()
      expect(bridge.name).toBe('strands:lifecycle-bridge')
    })

    it('has initAgent method', () => {
      const bridge = new LifecycleBridge()
      expect(typeof bridge.initAgent).toBe('function')
    })

    it('has drain method', () => {
      const bridge = new LifecycleBridge()
      expect(typeof bridge.drain).toBe('function')
    })
  })

  describe('lifecycle events with simple text response', () => {
    it('produces lifecycle events for a text-only agent turn', async () => {
      const bridge = await runTextTurn()
      const events = bridge.drain()
      expect(events.length).toBeGreaterThan(0)

      const eventTypes = events.map((e) => e.val.eventType)

      expect(eventTypes).toContain('initialized')
      expect(eventTypes).toContain('before-invocation')
      expect(eventTypes).toContain('before-model-call')
      expect(eventTypes).toContain('after-model-call')
      expect(eventTypes).toContain('message-added')
      expect(eventTypes).toContain('after-invocation')

      const initialized = events.find((e) => e.val.eventType === 'initialized')
      expect(initialized).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'initialized', toolUse: undefined, toolResult: undefined },
      })

      const beforeInvocation = events.find((e) => e.val.eventType === 'before-invocation')
      expect(beforeInvocation).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'before-invocation', toolUse: undefined, toolResult: undefined },
      })

      const beforeModelCall = events.find((e) => e.val.eventType === 'before-model-call')
      expect(beforeModelCall).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'before-model-call', toolUse: undefined, toolResult: undefined },
      })

      const afterModelCall = events.find((e) => e.val.eventType === 'after-model-call')
      expect(afterModelCall).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'after-model-call', toolUse: undefined, toolResult: undefined },
      })

      const messageAdded = events.find((e) => e.val.eventType === 'message-added')
      expect(messageAdded).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'message-added', toolUse: undefined, toolResult: undefined },
      })

      const afterInvocation = events.find((e) => e.val.eventType === 'after-invocation')
      expect(afterInvocation).toStrictEqual({
        tag: 'lifecycle',
        val: { eventType: 'after-invocation', toolUse: undefined, toolResult: undefined },
      })
    })

    it('non-tool events have undefined toolUse and toolResult', async () => {
      const bridge = await runTextTurn()
      const events = bridge.drain()
      for (const event of events) {
        expect(event.tag).toBe('lifecycle')
        expect(event.val.toolUse).toBeUndefined()
        expect(event.val.toolResult).toBeUndefined()
      }
    })
  })

  describe('drain clears queue', () => {
    it('first drain returns events, second drain returns empty array', async () => {
      const bridge = await runTextTurn()

      const first = bridge.drain()
      expect(first.length).toBeGreaterThan(0)

      const second = bridge.drain()
      expect(second).toStrictEqual([])
    })
  })

  describe('tool-related lifecycle events', () => {
    it('produces before-tool-call and after-tool-call events with serialized data', async () => {
      const bridge = new LifecycleBridge()
      const model = new MockMessageModel()

      model.addTurn({
        type: 'toolUseBlock',
        name: 'test_tool',
        toolUseId: 'tu-1',
        input: { query: 'test' },
      })
      model.addTurn({ type: 'textBlock', text: 'Done' })

      const tool = new FunctionTool({
        name: 'test_tool',
        description: 'A test tool',
        inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
        callback: () => [{ text: 'tool result' }],
      })

      const agent = new Agent({
        model,
        plugins: [bridge],
        tools: [tool],
        printer: false,
      })

      await agent.invoke('use the tool')

      const events = bridge.drain()

      const beforeToolCall = events.find((e) => e.val.eventType === 'before-tool-call')
      expect(beforeToolCall).toStrictEqual({
        tag: 'lifecycle',
        val: {
          eventType: 'before-tool-call',
          toolUse: expect.any(String),
          toolResult: undefined,
        },
      })

      const afterToolCall = events.find((e) => e.val.eventType === 'after-tool-call')
      expect(afterToolCall).toStrictEqual({
        tag: 'lifecycle',
        val: {
          eventType: 'after-tool-call',
          toolUse: expect.any(String),
          toolResult: expect.any(String),
        },
      })
    })
  })
})
