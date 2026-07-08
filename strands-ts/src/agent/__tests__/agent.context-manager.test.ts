import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { SlidingWindowConversationManager } from '../../conversation-manager/sliding-window-conversation-manager.js'
import { SummarizingConversationManager } from '../../conversation-manager/summarizing-conversation-manager.js'
import { ContextOffloader } from '../../vended-plugins/context-offloader/plugin.js'
import { InMemoryStorage } from '../../vended-plugins/context-offloader/storage.js'
import type { ConversationManager } from '../../conversation-manager/conversation-manager.js'

function internals(agent: Agent): any {
  return agent as any
}

function getConversationManager(agent: Agent): ConversationManager {
  return internals(agent)._conversationManager
}

function getPending(agent: Agent): any[] {
  return internals(agent)._pluginRegistry._pending
}

describe('Agent contextManager', () => {
  describe('when undefined (default)', () => {
    it('uses SlidingWindowConversationManager', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model })
      expect(getConversationManager(agent)).toBeInstanceOf(SlidingWindowConversationManager)
    })

    it('does not add ContextOffloader plugin', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model })
      const pending = getPending(agent)
      expect(pending.find((p: any) => p.name === 'strands:context-offloader')).toBeUndefined()
    })
  })

  describe('when "auto"', () => {
    it('uses SummarizingConversationManager', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto' })
      expect(getConversationManager(agent)).toBeInstanceOf(SummarizingConversationManager)
    })

    it('sets summaryRatio to 0.3', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto' })
      const conversationManager = getConversationManager(agent) as any
      expect(conversationManager._summaryRatio).toBe(0.3)
    })

    it('enables proactive compression at 0.85', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto' })
      const conversationManager = getConversationManager(agent) as any
      expect(conversationManager._compressionThreshold).toBe(0.85)
    })

    it('adds ContextOffloader plugin with benchmark defaults', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const agent = new Agent({ model, contextManager: 'auto' })
      const pending = getPending(agent)
      const offloader = pending.find((p: any) => p.name === 'strands:context-offloader') as any
      expect(offloader).toBeDefined()
      expect(offloader._maxResultTokens).toBe(1500)
      expect(offloader._previewTokens).toBe(750)
      expect(offloader._storage).toBeInstanceOf(InMemoryStorage)
    })
  })

  describe('coexistence with conversationManager', () => {
    it('respects user-provided conversationManager', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const userCm = new SlidingWindowConversationManager({ windowSize: 20 })
      const agent = new Agent({ model, contextManager: 'auto', conversationManager: userCm })
      expect(getConversationManager(agent)).toBe(userCm)
    })

    it('still adds ContextOffloader when user provides conversationManager', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const userCm = new SlidingWindowConversationManager({ windowSize: 20 })
      const agent = new Agent({ model, contextManager: 'auto', conversationManager: userCm })
      const pending = getPending(agent)
      expect(pending.find((p: any) => p.name === 'strands:context-offloader')).toBeDefined()
    })

    it('does not add duplicate ContextOffloader if user provides one', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const userOffloader = new ContextOffloader({
        storage: new InMemoryStorage(),
        maxResultTokens: 3000,
        previewTokens: 1000,
      })
      const agent = new Agent({ model, contextManager: 'auto', plugins: [userOffloader] })
      const pending = getPending(agent)
      const offloaders = pending.filter((p: any) => p.name === 'strands:context-offloader')
      expect(offloaders).toHaveLength(1)
      expect((offloaders[0] as any)._maxResultTokens).toBe(3000)
    })
  })

  describe('stateful model', () => {
    it('throws when used with a stateful model', () => {
      class StatefulModel extends MockMessageModel {
        override get stateful(): boolean {
          return true
        }
      }
      const model = new StatefulModel().addTurn({ type: 'textBlock', text: 'hi' })
      expect(() => new Agent({ model, contextManager: 'auto' })).toThrow('stateful model')
    })
  })

  describe('unsupported value', () => {
    it('throws for invalid contextManager value', () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      expect(() => new Agent({ model, contextManager: 'manual' as any })).toThrow('Unsupported contextManager value')
    })
  })
})
