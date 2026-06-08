import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { SlidingWindowConversationManager } from '../../conversation-manager/sliding-window-conversation-manager.js'
import { SummarizingConversationManager } from '../../conversation-manager/summarizing-conversation-manager.js'
import { ContextOffloader } from '../../vended-plugins/context-offloader/plugin.js'
import { InMemoryStorage } from '../../vended-plugins/context-offloader/storage.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { LocalAgent } from '../../types/agent.js'

type AgentInternals = {
  _conversationManager: { _summaryRatio?: number; _compressionThreshold?: number }
  _pluginRegistry: { _plugins: Map<string, Plugin>; _pending: Plugin[] }
}

function getInternals(agent: Agent): AgentInternals {
  return agent as unknown as AgentInternals
}

class StatefulMockModel extends MockMessageModel {
  override get stateful(): boolean {
    return true
  }
}

describe('Agent contextManager option', () => {
  describe('when not set', () => {
    it('preserves default SlidingWindowConversationManager', () => {
      const agent = new Agent({ model: new MockMessageModel(), printer: false })
      expect(getInternals(agent)._conversationManager).toBeInstanceOf(SlidingWindowConversationManager)
    })

    it('does not add ContextOffloader plugin', () => {
      const agent = new Agent({ model: new MockMessageModel(), printer: false })
      const { _plugins, _pending } = getInternals(agent)._pluginRegistry
      const allNames = [..._plugins.keys(), ..._pending.map((p) => p.name)]
      expect(allNames).not.toContain('strands:context-offloader')
    })
  })

  describe('when set to "auto"', () => {
    it('uses SummarizingConversationManager', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      expect(getInternals(agent)._conversationManager).toBeInstanceOf(SummarizingConversationManager)
    })

    it('configures summaryRatio to 0.3', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      expect(getInternals(agent)._conversationManager._summaryRatio).toBe(0.3)
    })

    it('configures proactive compression threshold to 0.85', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      expect(getInternals(agent)._conversationManager._compressionThreshold).toBe(0.85)
    })

    it('adds ContextOffloader plugin', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloader = _pending.find((p) => p.name === 'strands:context-offloader')
      expect(offloader).toBeDefined()
      expect(offloader).toBeInstanceOf(ContextOffloader)
    })

    it('configures offloader maxResultTokens to 1500', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloader = _pending.find((p) => p.name === 'strands:context-offloader') as unknown as {
        _maxResultTokens: number
      }
      expect(offloader._maxResultTokens).toBe(1500)
    })

    it('configures offloader previewTokens to 750', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloader = _pending.find((p) => p.name === 'strands:context-offloader') as unknown as {
        _previewTokens: number
      }
      expect(offloader._previewTokens).toBe(750)
    })

    it('configures offloader with InMemoryStorage', () => {
      const agent = new Agent({ model: new MockMessageModel(), contextManager: 'auto', printer: false })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloader = _pending.find((p) => p.name === 'strands:context-offloader') as unknown as {
        _storage: unknown
      }
      expect(offloader._storage).toBeInstanceOf(InMemoryStorage)
    })
  })

  describe('coexistence with user options', () => {
    it('respects user-provided conversationManager', () => {
      const userCm = new SlidingWindowConversationManager({ windowSize: 20 })
      const agent = new Agent({
        model: new MockMessageModel(),
        contextManager: 'auto',
        conversationManager: userCm,
        printer: false,
      })
      expect(getInternals(agent)._conversationManager).toBe(userCm)
    })

    it('still adds ContextOffloader when user provides conversationManager', () => {
      const userCm = new SlidingWindowConversationManager({ windowSize: 20 })
      const agent = new Agent({
        model: new MockMessageModel(),
        contextManager: 'auto',
        conversationManager: userCm,
        printer: false,
      })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloader = _pending.find((p) => p.name === 'strands:context-offloader')
      expect(offloader).toBeDefined()
    })

    it('does not duplicate user-provided ContextOffloader', () => {
      const userOffloader = new ContextOffloader({
        storage: new InMemoryStorage(),
        maxResultTokens: 3000,
        previewTokens: 1000,
      })
      const agent = new Agent({
        model: new MockMessageModel(),
        contextManager: 'auto',
        plugins: [userOffloader],
        printer: false,
      })
      const { _pending } = getInternals(agent)._pluginRegistry
      const offloaders = _pending.filter((p) => p.name === 'strands:context-offloader')
      expect(offloaders).toHaveLength(1)
      expect((offloaders[0] as unknown as { _maxResultTokens: number })._maxResultTokens).toBe(3000)
    })

    it('preserves user plugins alongside auto-configured offloader', () => {
      const customPlugin: Plugin = {
        name: 'my-plugin',
        initAgent(_agent: LocalAgent) {},
      }
      const agent = new Agent({
        model: new MockMessageModel(),
        contextManager: 'auto',
        plugins: [customPlugin],
        printer: false,
      })
      const { _pending } = getInternals(agent)._pluginRegistry
      const names = _pending.map((p) => p.name)
      expect(names).toContain('my-plugin')
      expect(names).toContain('strands:context-offloader')
    })
  })

  describe('with stateful model', () => {
    it('throws when contextManager is set', () => {
      expect(() => new Agent({ model: new StatefulMockModel(), contextManager: 'auto', printer: false })).toThrow(
        /stateful model/
      )
    })
  })
})
