import { describe, it, expect } from 'vitest'
import { ContextManager } from '../context-manager.js'
import { Manifest } from '../manifest.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { Message, TextBlock } from '../../types/messages.js'
import { createMockAgent, invokeTrackedHook } from '../../__fixtures__/agent-helpers.js'

function makeMockAgent(overrides?: { id?: string }) {
  const agent = createMockAgent()
  Object.defineProperty(agent, 'id', { value: overrides?.id ?? 'test-agent', writable: false })
  return agent
}

function makeMessage(text: string, trackingId?: string): Message {
  return new Message(
    trackingId
      ? { role: 'user', content: [new TextBlock(text)], trackingId }
      : { role: 'user', content: [new TextBlock(text)] }
  )
}

describe('ContextManager', () => {
  describe('constructor', () => {
    it('uses InMemoryStorage internally', () => {
      const cm = new ContextManager()
      expect(cm.stashEnabled).toBe(true)
    })

    it('stash enabled by default', () => {
      const cm = new ContextManager()
      expect(cm.stashEnabled).toBe(true)
    })

    it('stash disabled with false', () => {
      const cm = new ContextManager({ stash: false })
      expect(cm.stashEnabled).toBe(false)
    })

    it('stash disabled with config object', () => {
      const cm = new ContextManager({ stash: { enabled: false } })
      expect(cm.stashEnabled).toBe(false)
    })

    it('stash enabled with config object', () => {
      const cm = new ContextManager({ stash: { enabled: true } })
      expect(cm.stashEnabled).toBe(true)
    })

    it('has correct plugin name', () => {
      const cm = new ContextManager()
      expect(cm.name).toBe('strands:context-manager')
    })
  })

  describe('initAgent', () => {
    it('creates stash when stash enabled', () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)
      expect(cm.stash).toBeDefined()
    })

    it('does not create stash when stash disabled', () => {
      const cm = new ContextManager({ stash: false })
      const agent = makeMockAgent()
      cm.initAgent(agent)
      expect(cm.stash).toBeUndefined()
    })

    it('registers MessageAddedEvent hook', () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)
      const hook = agent.trackedHooks.find((h) => h.eventType === MessageAddedEvent)
      expect(hook).toBeDefined()
    })
  })

  describe('L1 write on arrival', () => {
    it('writes message to storage via hook', async () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)

      const message = makeMessage('hello world', 'msg-100')
      await invokeTrackedHook(agent, new MessageAddedEvent({ agent, message, invocationState: {} }))

      const manifest = await cm.stash!.getManifest()
      expect(manifest.has('msg-100')).toBe(true)
    })

    it('duplicate message is idempotent', async () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)

      const message = makeMessage('hello', 'msg-200')
      await invokeTrackedHook(agent, new MessageAddedEvent({ agent, message, invocationState: {} }))
      await invokeTrackedHook(agent, new MessageAddedEvent({ agent, message, invocationState: {} }))

      const manifest = await cm.stash!.getManifest()
      expect(manifest.entries).toHaveLength(1)
    })

    it('stash disabled skips write without error', async () => {
      const cm = new ContextManager({ stash: false })
      const agent = makeMockAgent()
      cm.initAgent(agent)

      const message = makeMessage('dropped', 'msg-300')
      await invokeTrackedHook(agent, new MessageAddedEvent({ agent, message, invocationState: {} }))
    })
  })

  describe('stash read/write', () => {
    it('round-trips a message through storage', async () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)

      const message = new Message({ role: 'assistant', content: [new TextBlock('response')], trackingId: 'msg-400' })
      await cm.stash!.writeMessage(message)

      const readBack = await cm.stash!.readMessage('msg-400')
      expect(readBack).not.toBeNull()
      expect(readBack!['role']).toBe('assistant')
      expect(readBack!['trackingId']).toBe('msg-400')
    })

    it('returns null for nonexistent message', async () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)

      const result = await cm.stash!.readMessage('nonexistent')
      expect(result).toBeNull()
    })

    it('tracks multiple messages in manifest', async () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)

      for (let idx = 0; idx < 5; idx++) {
        const role = idx % 2 === 0 ? 'user' : 'assistant'
        const message = new Message({ role, content: [new TextBlock(`msg ${idx}`)], trackingId: `msg-${idx}` })
        await cm.stash!.writeMessage(message)
      }

      const manifest = await cm.stash!.getManifest()
      expect(manifest.entries).toHaveLength(5)
      expect(manifest.entries[0]!.trackingId).toBe('msg-0')
      expect(manifest.entries[4]!.trackingId).toBe('msg-4')
    })
  })

  describe('Manifest', () => {
    it('serialize/deserialize roundtrip', () => {
      const manifest = new Manifest()
      manifest.add({ trackingId: 'a', role: 'user', storageKey: 'a', contentBlocks: 1 })
      manifest.add({ trackingId: 'b', role: 'assistant', storageKey: 'b', contentBlocks: 2 })

      const data = manifest.serialize()
      const restored = Manifest.deserialize(data)

      expect(restored.entries).toHaveLength(2)
      expect(restored.entries[0]!.trackingId).toBe('a')
      expect(restored.entries[1]!.trackingId).toBe('b')
      expect(restored.entries[1]!.contentBlocks).toBe(2)
    })

    it('has() returns correct results', () => {
      const manifest = new Manifest()
      manifest.add({ trackingId: 'x', role: 'user', storageKey: 'x', contentBlocks: 0 })
      expect(manifest.has('x')).toBe(true)
      expect(manifest.has('y')).toBe(false)
    })
  })
})
