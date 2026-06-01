import { describe, it, expect } from 'vitest'
import { isProtected, pinMessage } from '../protection.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../../types/messages.js'

function userMsg(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function assistantMsg(text: string): Message {
  return new Message({ role: 'assistant', content: [new TextBlock(text)] })
}

function toolUseMsg(toolUseId: string): Message {
  return new Message({ role: 'assistant', content: [new ToolUseBlock({ toolUseId, name: 'test', input: {} })] })
}

function toolResultMsg(toolUseId: string): Message {
  return new Message({
    role: 'user',
    content: [new ToolResultBlock({ toolUseId, content: [new TextBlock('result')], status: 'success' })],
  })
}

describe('isProtected', () => {
  describe('no range, no pin', () => {
    it('returns false for unprotected message', () => {
      const messages = [userMsg('a'), assistantMsg('b')]
      expect(isProtected(messages, 0)).toBe(false)
      expect(isProtected(messages, 1)).toBe(false)
    })
  })

  describe('positive range (protect first N)', () => {
    it('protects messages within range', () => {
      const messages = [userMsg('a'), assistantMsg('b'), userMsg('c')]
      expect(isProtected(messages, 0, 2)).toBe(true)
      expect(isProtected(messages, 1, 2)).toBe(true)
      expect(isProtected(messages, 2, 2)).toBe(false)
    })

    it('protects toolUse outside range if its toolResult is inside range', () => {
      const messages = [userMsg('task'), toolUseMsg('t1'), toolResultMsg('t1'), userMsg('next')]
      // range=2 protects [0] and [1]. [2] is toolResult — check if toolUse at [1] being in range protects [2]
      // Actually [2] is outside range. But [1] (toolUse) is in range, so [2] (toolResult, partner) should be protected.
      expect(isProtected(messages, 2, 2)).toBe(true)
    })

    it('protects toolResult outside range if its toolUse is inside range', () => {
      const messages = [toolUseMsg('t1'), toolResultMsg('t1'), userMsg('a'), assistantMsg('b')]
      // range=1 protects [0] (toolUse). [1] (toolResult) is outside but partner is protected.
      expect(isProtected(messages, 1, 1)).toBe(true)
    })
  })

  describe('pinned messages', () => {
    it('protects pinned message regardless of range', () => {
      const messages = [userMsg('a'), pinMessage(assistantMsg('pinned')), userMsg('c')]
      expect(isProtected(messages, 1)).toBe(true)
      expect(isProtected(messages, 1, 0)).toBe(true)
    })

    it('protects tool-pair partner of pinned message', () => {
      const messages = [pinMessage(toolUseMsg('t1')), toolResultMsg('t1'), userMsg('a')]
      expect(isProtected(messages, 1)).toBe(true) // toolResult partner of pinned toolUse
    })
  })
})
