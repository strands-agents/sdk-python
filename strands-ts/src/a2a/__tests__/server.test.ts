import { describe, expect, it, vi } from 'vitest'
import { A2AServer } from '../server.js'
import { Agent } from '../../agent/agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'

describe('A2AServer', () => {
  describe('constructor', () => {
    it('builds agent card with provided values', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const server = new A2AServer({
        agent: new Agent({ model, printer: false }),
        name: 'Base Agent',
        description: 'A base agent',
        httpUrl: 'http://example.com',
        version: '2.0.0',
      })

      expect(server.agentCard).toStrictEqual({
        name: 'Base Agent',
        description: 'A base agent',
        version: '2.0.0',
        protocolVersion: '0.2.0',
        url: 'http://example.com',
        defaultInputModes: ['text/plain'],
        defaultOutputModes: ['text/plain'],
        skills: [],
        capabilities: { streaming: true },
      })
    })

    it('uses default values when optional config is omitted', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const server = new A2AServer({
        agent: new Agent({ model, printer: false }),
        name: 'Minimal Agent',
      })

      expect(server.agentCard.description).toBe('')
      expect(server.agentCard.version).toBe('0.0.1')
      expect(server.agentCard.url).toBe('')
      expect(server.agentCard.skills).toStrictEqual([])
    })

    it('accepts custom taskStore', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const taskStore = { save: vi.fn(), load: vi.fn() }
      const server = new A2AServer({
        agent: new Agent({ model, printer: false }),
        name: 'Agent',
        taskStore,
      })
      expect(server.agentCard).toBeDefined()
    })
  })
})
