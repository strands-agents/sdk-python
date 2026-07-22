import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { SessionManager } from '../../session/session-manager.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'

describe('Agent', () => {
  describe('sessionId', () => {
    it('uses explicit sessionId from config', () => {
      const agent = new Agent({ model: new MockMessageModel(), sessionId: 'my-session' })

      expect(agent.sessionId).toBe('my-session')
    })

    it('inherits sessionId from SessionManager when not provided', () => {
      const sessionManager = new SessionManager({ sessionId: 'sm-session', storage: new InMemoryStorage() })
      const agent = new Agent({ model: new MockMessageModel(), sessionManager })

      expect(agent.sessionId).toBe('sm-session')
    })

    it('prefers explicit sessionId over SessionManager sessionId', () => {
      const sessionManager = new SessionManager({ sessionId: 'sm-session', storage: new InMemoryStorage() })
      const agent = new Agent({ model: new MockMessageModel(), sessionId: 'explicit-session', sessionManager })

      expect(agent.sessionId).toBe('explicit-session')
    })

    it('auto-generates a UUID when no sessionId or SessionManager is provided', () => {
      const agent = new Agent({ model: new MockMessageModel() })

      expect(agent.sessionId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    })

    it('generates unique sessionIds for different agent instances', () => {
      const agent1 = new Agent({ model: new MockMessageModel() })
      const agent2 = new Agent({ model: new MockMessageModel() })

      expect(agent1.sessionId).not.toBe(agent2.sessionId)
    })
  })

  describe('duplicate agent id check', () => {
    it('throws when two agents share the same SessionManager and agentId', () => {
      const sessionManager = new SessionManager({ sessionId: 'shared-session', storage: new InMemoryStorage() })
      new Agent({ model: new MockMessageModel(), id: 'researcher', sessionManager })

      expect(() => new Agent({ model: new MockMessageModel(), id: 'researcher', sessionManager })).toThrow(
        'an agent with this id already exists in this session'
      )
    })

    it('allows different agent ids in the same session', () => {
      const sessionManager = new SessionManager({ sessionId: 'shared-session', storage: new InMemoryStorage() })
      new Agent({ model: new MockMessageModel(), id: 'researcher', sessionManager })

      expect(() => new Agent({ model: new MockMessageModel(), id: 'writer', sessionManager })).not.toThrow()
    })

    it('allows same agent id across different SessionManager instances', () => {
      const sessionManager1 = new SessionManager({ sessionId: 'session-1', storage: new InMemoryStorage() })
      const sessionManager2 = new SessionManager({ sessionId: 'session-2', storage: new InMemoryStorage() })
      new Agent({ model: new MockMessageModel(), id: 'researcher', sessionManager: sessionManager1 })

      expect(
        () => new Agent({ model: new MockMessageModel(), id: 'researcher', sessionManager: sessionManager2 })
      ).not.toThrow()
    })

    it('does not check when no SessionManager is configured', () => {
      new Agent({ model: new MockMessageModel(), id: 'agent', sessionId: 'same-session' })

      expect(() => new Agent({ model: new MockMessageModel(), id: 'agent', sessionId: 'same-session' })).not.toThrow()
    })
  })
})
