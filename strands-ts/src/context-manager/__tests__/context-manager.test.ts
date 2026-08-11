import { describe, it, expect } from 'vitest'
import { ContextManager } from '../context-manager.js'
import { createMockAgent } from '../../__fixtures__/agent-helpers.js'

function makeMockAgent(overrides?: { id?: string }) {
  const agent = createMockAgent()
  Object.defineProperty(agent, 'id', { value: overrides?.id ?? 'test-agent', writable: false })
  Object.defineProperty(agent, 'model', {
    value: { getConfig: () => ({}), countTokens: async () => 0, estimateUtilization: () => 0 },
    writable: false,
  })
  return agent
}

describe('ContextManager', () => {
  describe('constructor', () => {
    it('has correct plugin name', () => {
      const cm = new ContextManager()
      expect(cm.name).toBe('strands:context-manager')
    })

    it('accepts custom strategies', () => {
      const cm = new ContextManager({
        strategies: [{ name: 'custom', apply: async () => false }],
      })
      expect(cm).toBeDefined()
    })
  })

  describe('initAgent', () => {
    it('registers AfterModelCallEvent hook', () => {
      const cm = new ContextManager()
      const agent = makeMockAgent()
      cm.initAgent(agent)
      expect(agent.trackedHooks.length).toBeGreaterThan(0)
    })

    it('initializes strategies with init context', () => {
      let initCalled = false
      const strategy = {
        name: 'test',
        init: () => {
          initCalled = true
        },
        apply: async () => false,
      }
      const cm = new ContextManager({ strategies: [strategy] })
      const agent = makeMockAgent()
      cm.initAgent(agent)
      expect(initCalled).toBe(true)
    })
  })
})
