import { describe, it, expect } from 'vitest'
import { truncate } from '../truncate.js'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '../../../../types/messages.js'
import { pinMessage } from '../../protection.js'

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

describe('truncate', () => {
  it('removes oldest messages when over window size', () => {
    const messages = [
      userMsg('Message 1'),
      assistantMsg('Response 1'),
      userMsg('Message 2'),
      assistantMsg('Response 2'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
    ]

    const result = truncate(messages, 4)

    expect(result).toBe(true)
    expect(messages).toHaveLength(4)
    expect((messages[0]!.content[0] as TextBlock).text).toBe('Message 2')
  })

  it('preserves tool use/result pairs', () => {
    const messages = [
      userMsg('Message 1'),
      toolUseMsg('t1'),
      toolResultMsg('t1'),
      userMsg('Message 2'),
      assistantMsg('Response 2'),
    ]

    // Window size 2, trimIndex starts at 3 (5 - 2 = 3), which is user 'Message 2' - valid
    const result = truncate(messages, 2)

    expect(result).toBe(true)
    // Should trim at a point that doesn't break tool use/result pairs
    expect(messages[0]!.role).toBe('user')
  })

  it('returns false when messages.length <= 2', () => {
    const messages = [userMsg('Message 1'), assistantMsg('Response 1')]

    const result = truncate(messages, 1)

    expect(result).toBe(false)
    expect(messages).toHaveLength(2)
  })

  it('returns false for single message', () => {
    const messages = [userMsg('only')]

    const result = truncate(messages, 0)

    expect(result).toBe(false)
    expect(messages).toHaveLength(1)
  })

  it('respects protectFirst', () => {
    const messages = [
      userMsg('Protected 1'),
      assistantMsg('Protected 2'),
      userMsg('Message 3'),
      assistantMsg('Response 3'),
      userMsg('Message 4'),
      assistantMsg('Response 4'),
    ]

    const result = truncate(messages, 3, { protectFirst: 2 })

    expect(result).toBe(true)
    // First 2 messages should still be there
    expect((messages[0]!.content[0] as TextBlock).text).toBe('Protected 1')
    expect((messages[1]!.content[0] as TextBlock).text).toBe('Protected 2')
  })

  it('returns false when all messages in trim range are protected', () => {
    const messages = [
      pinMessage(userMsg('Pinned 1')),
      pinMessage(assistantMsg('Pinned 2')),
      pinMessage(userMsg('Pinned 3')),
      assistantMsg('Response'),
      userMsg('Last'),
    ]

    // trimIndex will be in range where all are pinned
    const result = truncate(messages, 4)

    // The trim range [0, trimIndex) only contains pinned messages
    // trimIndex = max(2, 5-4) = 2, range [0,2) are pinned -> returns false...
    // Actually trimIndex = max(2, 5-4=1) = 2, but findValidTrimPoint may adjust.
    // Let's see: trimIndex = max(messages.length - windowSize, 2) = max(1, 2) = 2
    // findValidTrimPoint(messages, 2): messages[2] is pinned user, role=user, no toolResult.
    // So trimIndex = 2, range [0,2) -> indices 0,1 are pinned -> returns false
    expect(result).toBe(false)
  })

  it('does not orphan toolResult at trim boundary', () => {
    const messages = [
      userMsg('Start'),
      toolUseMsg('t1'),
      toolResultMsg('t1'),
      userMsg('After tools'),
      assistantMsg('Response'),
    ]

    // windowSize=2, trimIndex = 5-2=3, message[3] is user 'After tools', no toolResult -> valid
    const result = truncate(messages, 2)

    expect(result).toBe(true)
    // First message after trim should not be a tool result
    const firstMsg = messages[0]!
    const hasToolResult = firstMsg.content.some((b) => b.type === 'toolResultBlock')
    expect(hasToolResult).toBe(false)
  })

  it('skips assistant messages to find valid user trim point', () => {
    const messages = [
      userMsg('Message 1'),
      assistantMsg('Response 1'),
      assistantMsg('Response 2'), // Non-user at potential trim index
      userMsg('Message 2'),
      assistantMsg('Response 3'),
    ]

    // windowSize=2, trimIndex = 5-2=3, messages[3] is user 'Message 2' -> valid
    const result = truncate(messages, 2)

    expect(result).toBe(true)
    expect(messages[0]!.role).toBe('user')
    expect((messages[0]!.content[0] as TextBlock).text).toBe('Message 2')
  })
})
