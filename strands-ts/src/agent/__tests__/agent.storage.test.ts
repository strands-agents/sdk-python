import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'
import { NAMESPACED } from '../../storage/storage.js'
import { ContextOffloader } from '../../vended-plugins/context-offloader/plugin.js'
import { SessionManager } from '../../session/session-manager.js'

function internals(agent: Agent): any {
  return agent as any
}

describe('Agent storage', () => {
  describe('agent-level storage flows to ContextOffloader', () => {
    it('auto-created offloader uses agent storage when provided', async () => {
      const storage = new InMemoryStorage()
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto', storage })
      await agent.invoke('hi')
      const plugins = internals(agent)._pluginRegistry._plugins
      const offloader = plugins.get('strands:context-offloader') as any
      expect(offloader._storage).toBeDefined()
      expect(NAMESPACED in offloader._storage).toBe(true)

      await offloader._storage.write('test-key', new TextEncoder().encode('test-value'))
      const stored = await storage.read('offloader/test-key')
      expect(stored).not.toBeNull()
      expect(new TextDecoder().decode(stored!)).toBe('test-value')
    })

    it('auto-created offloader falls back to InMemoryStorage when no agent storage', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto' })
      await agent.invoke('hi')
      const plugins = internals(agent)._pluginRegistry._plugins
      const offloader = plugins.get('strands:context-offloader') as any
      expect(offloader._storage).toBeDefined()
      expect(NAMESPACED in offloader._storage).toBe(true)
    })

    it('explicit offloader storage overrides agent storage', async () => {
      const agentStorage = new InMemoryStorage()
      const offloaderStorage = new InMemoryStorage()
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({
        model,
        storage: agentStorage,
        plugins: [new ContextOffloader({ storage: offloaderStorage, maxResultTokens: 1500, previewTokens: 750 })],
        contextManager: 'auto',
      })
      await agent.invoke('hi')
      const plugins = internals(agent)._pluginRegistry._plugins
      const offloader = plugins.get('strands:context-offloader') as any

      await offloader._storage.write('test-key', new TextEncoder().encode('test-value'))
      const inOffloader = await offloaderStorage.read('offloader/test-key')
      expect(inOffloader).not.toBeNull()
      const inAgent = await agentStorage.read('offloader/test-key')
      expect(inAgent).toBeNull()
    })
  })

  describe('agent-level storage flows to SessionManager', () => {
    it('session manager resolves storage from agent when not explicitly provided', async () => {
      const storage = new InMemoryStorage()
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const sessionManager = new SessionManager({ sessionId: 'test-session' })
      const agent = new Agent({ model, storage, sessionManager })
      await agent.invoke('hi')

      const keys = await storage.list('session/')
      expect(keys.length).toBeGreaterThan(0)
    })

    it('explicit session manager storage overrides agent storage', async () => {
      const agentStorage = new InMemoryStorage()
      const sessionStorage = new InMemoryStorage()
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const sessionManager = new SessionManager({ sessionId: 'test-session', storage: sessionStorage })
      const agent = new Agent({ model, storage: agentStorage, sessionManager })
      await agent.invoke('hi')

      const sessionKeys = await sessionStorage.list('session/')
      expect(sessionKeys.length).toBeGreaterThan(0)
      const agentKeys = await agentStorage.list('session/')
      expect(agentKeys.length).toBe(0)
    })

    it('throws when session manager has no storage and agent has no storage', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const sessionManager = new SessionManager({ sessionId: 'test-session' })
      const agent = new Agent({ model, sessionManager })
      await expect(agent.invoke('hi')).rejects.toThrow('SessionManager requires a storage backend')
    })
  })

  describe('agent.storage property', () => {
    it('returns the configured storage', () => {
      const storage = new InMemoryStorage()
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, storage })
      expect(agent.storage).toBe(storage)
    })

    it('returns undefined when no storage configured', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model })
      expect(agent.storage).toBeUndefined()
    })
  })
})
