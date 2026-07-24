import { describe, it, expect } from 'vitest'
import { stop, makeStop } from '../index.js'
import { DEFAULT_STOP_MESSAGE, DEFAULT_MAX_STOP_MESSAGE_LENGTH, STOP_INVOCATION_STATE_KEY } from '../types.js'
import type { ToolContext } from '../../../index.js'
import { AfterToolsEvent } from '../../../hooks/events.js'
import { Message } from '../../../types/messages.js'
import { createMockAgent, invokeTrackedHook, type MockAgent } from '../../../__fixtures__/agent-helpers.js'

const createFreshContext = (
  invocationState: Record<string, unknown> = {}
): { agent: MockAgent; context: ToolContext } => {
  const agent = createMockAgent()
  const context: ToolContext = {
    toolUse: { name: 'stop', toolUseId: 'test-id', input: {} },
    agent,
    invocationState,
    interrupt: () => {
      throw new Error('interrupt not available in mock context')
    },
  }
  return { agent, context }
}

describe('stop tool', () => {
  describe('behavior', () => {
    it('returns the provided message verbatim', async () => {
      const { context } = createFreshContext()
      const result = await stop.invoke({ message: 'all done' }, context)
      expect(result).toBe('all done')
    })

    it('returns the default message when none is provided', async () => {
      const { context } = createFreshContext()
      const result = await stop.invoke({}, context)
      expect(result).toBe(DEFAULT_STOP_MESSAGE)
    })

    it('falls back to the default when message is an empty string, so the loop still halts', async () => {
      // The agent loop treats `endTurn === ''` as falsy, so an empty-string
      // message must not short-circuit the stop signal.
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      const result = await stop.invoke({ message: '' }, context)
      expect(result).toBe(DEFAULT_STOP_MESSAGE)

      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)
      expect(event.endTurn).toBe(DEFAULT_STOP_MESSAGE)
    })

    it('falls back to the default when message is null, so providers that serialize omitted fields as null still halt', async () => {
      // Some providers serialize an omitted optional field as JSON null rather
      // than absent; the schema must accept null and the callback must treat
      // it identically to undefined.
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      const result = await stop.invoke({ message: null as unknown as string }, context)
      expect(result).toBe(DEFAULT_STOP_MESSAGE)

      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)
      expect(event.endTurn).toBe(DEFAULT_STOP_MESSAGE)
    })

    it('installs exactly one AfterToolsEvent hook per agent, even across multiple calls', async () => {
      const { agent, context } = createFreshContext()
      await stop.invoke({ message: 'first' }, context)
      await stop.invoke({ message: 'second' }, context)
      const hooks = agent.trackedHooks.filter((h) => h.eventType === AfterToolsEvent)
      expect(hooks).toHaveLength(1)
    })

    it('routes the message through AfterToolsEvent.endTurn', async () => {
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      await stop.invoke({ message: 'finished' }, context)

      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)

      expect(event.endTurn).toBe('finished')
    })

    it('falls back to the default message when routing through the hook without one', async () => {
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      await stop.invoke({}, context)

      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)

      expect(event.endTurn).toBe(DEFAULT_STOP_MESSAGE)
    })

    it('leaves endTurn untouched when the tool was not called in this batch', async () => {
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      // First invocation to install the hook on this agent.
      await stop.invoke({ message: 'first turn' }, context)

      // Simulate that the marker was consumed by the previous batch's hook
      // firing, and now a fresh invocation runs without calling stop.
      delete invocationState[STOP_INVOCATION_STATE_KEY]
      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)

      expect(event.endTurn).toBe(false)
    })

    it('treats a null marker as absent so a JSON-round-tripped invocationState does not falsely fire', async () => {
      // If `invocationState` is serialized (e.g. persisted across turns) the
      // marker key may come back as null rather than missing. Both must be
      // treated as "no stop requested".
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      await stop.invoke({ message: 'install hook' }, context)

      invocationState[STOP_INVOCATION_STATE_KEY] = null
      const event = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, event)

      expect(event.endTurn).toBe(false)
    })

    it('consumes the marker after firing so the same invocationState does not re-trigger', async () => {
      const invocationState: Record<string, unknown> = {}
      const { agent, context } = createFreshContext(invocationState)
      await stop.invoke({ message: 'once' }, context)

      const first = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, first)
      expect(first.endTurn).toBe('once')

      const second = new AfterToolsEvent({
        agent,
        message: new Message({ role: 'user', content: [] }),
        invocationState,
      })
      await invokeTrackedHook(agent, second)
      expect(second.endTurn).toBe(false)
    })
  })

  describe('input validation', () => {
    it('rejects an oversized message', async () => {
      const { context } = createFreshContext()
      const oversized = 'x'.repeat(DEFAULT_MAX_STOP_MESSAGE_LENGTH + 1)
      await expect(stop.invoke({ message: oversized }, context)).rejects.toThrow(/maximum/i)
    })

    it('accepts a message at the length cap', async () => {
      const { context } = createFreshContext()
      const atCap = 'x'.repeat(DEFAULT_MAX_STOP_MESSAGE_LENGTH)
      const result = await stop.invoke({ message: atCap }, context)
      expect(result).toBe(atCap)
    })

    it('rejects a non-string message', async () => {
      const { context } = createFreshContext()
      await expect(stop.invoke({ message: 123 as unknown as string }, context)).rejects.toThrow()
    })

    it('does not set the marker when validation fails', async () => {
      const invocationState: Record<string, unknown> = {}
      const { context } = createFreshContext(invocationState)
      const oversized = 'x'.repeat(DEFAULT_MAX_STOP_MESSAGE_LENGTH + 1)
      await expect(stop.invoke({ message: oversized }, context)).rejects.toThrow()
      expect(invocationState).not.toHaveProperty(STOP_INVOCATION_STATE_KEY)
    })

    it('throws when invoked without a tool context', async () => {
      await expect(stop.invoke({})).rejects.toThrow(/context is required/i)
    })

    it('relaxes the cap when maxMessageLength is configured', async () => {
      const { context } = createFreshContext()
      const bigStop = makeStop({ maxMessageLength: 10_000 })
      const message = 'x'.repeat(8000)
      const result = await bigStop.invoke({ message }, context)
      expect(result).toBe(message)
    })

    it('still enforces the configured cap when it is exceeded', async () => {
      const { context } = createFreshContext()
      const bigStop = makeStop({ maxMessageLength: 10_000 })
      await expect(bigStop.invoke({ message: 'x'.repeat(10_001) }, context)).rejects.toThrow(/maximum of 10000/)
    })

    it('rejects a non-positive maxMessageLength at factory time', () => {
      expect(() => makeStop({ maxMessageLength: 0 })).toThrow(/positive integer/)
      expect(() => makeStop({ maxMessageLength: -1 })).toThrow(/positive integer/)
      expect(() => makeStop({ maxMessageLength: 1.5 })).toThrow(/positive integer/)
    })
  })

  describe('metadata', () => {
    it('supports a custom name', () => {
      const finish = makeStop({ name: 'finish' })
      expect(finish.name).toBe('finish')
    })

    it('supports a custom description', () => {
      const custom = makeStop({ description: 'custom desc' })
      expect(custom.description).toBe('custom desc')
    })

    it('exposes message as an optional property in the tool spec', () => {
      const schema = stop.toolSpec.inputSchema as { properties?: Record<string, unknown>; required?: string[] }
      expect(schema.properties).toHaveProperty('message')
      expect(schema.required ?? []).not.toContain('message')
    })
  })
})
