import { describe, it, expect, vi } from 'vitest'
import { SlidingWindowConversationManager } from '../sliding-window-conversation-manager.js'
import {
  ContextWindowOverflowError,
  DocumentBlock,
  ImageBlock,
  JsonBlock,
  Message,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  VideoBlock,
  type Model,
} from '../../index.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../../hooks/events.js'
import { createMockAgent, invokeTrackedHook } from '../../__fixtures__/agent-helpers.js'
import type { Agent } from '../../agent/agent.js'
import type { BaseModelConfig } from '../../models/model.js'

async function triggerSlidingWindow(manager: SlidingWindowConversationManager, agent: Agent): Promise<void> {
  const pluginAgent = createMockAgent()
  manager.initAgent(pluginAgent)
  await invokeTrackedHook(pluginAgent, new AfterInvocationEvent({ agent, invocationState: {} }))
}

// Helper to trigger context overflow handling through hooks
async function triggerContextOverflow(
  manager: SlidingWindowConversationManager,
  agent: Agent,
  error: Error
): Promise<{ retry?: boolean }> {
  const pluginAgent = createMockAgent()
  manager.initAgent(pluginAgent)
  const event = new AfterModelCallEvent({ agent, model: {} as any, attemptCount: 1, error, invocationState: {} })
  await invokeTrackedHook(pluginAgent, event)
  return event
}

describe('SlidingWindowConversationManager', () => {
  describe('constructor', () => {
    it('sets default windowSize to 40', () => {
      const manager = new SlidingWindowConversationManager()
      // Access through type assertion since these are private
      expect((manager as any)._windowSize).toBe(40)
    })

    it('sets default shouldTruncateResults to true', () => {
      const manager = new SlidingWindowConversationManager()
      expect((manager as any)._shouldTruncateResults).toBe(true)
    })

    it('accepts custom windowSize', () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 20 })
      expect((manager as any)._windowSize).toBe(20)
    })

    it('accepts custom shouldTruncateResults', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: false })
      expect((manager as any)._shouldTruncateResults).toBe(false)
    })
  })

  describe('reduce', () => {
    it('returns true when tool results are truncated even though message count is unchanged', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock('x'.repeat(500))],
            }),
          ],
        }),
      ]

      const result = manager.reduce({
        agent: createMockAgent({ messages }),
        model: {} as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      expect(messages).toHaveLength(1) // length unchanged, but truncation occurred
    })

    it('returns true when messages are trimmed', () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]

      const result = manager.reduce({
        agent: createMockAgent({ messages }),
        model: {} as Model,
        error: new ContextWindowOverflowError('overflow'),
      })

      expect(result).toBe(true)
      expect(messages).toHaveLength(2)
    })
  })

  describe('applyManagement', () => {
    it('skips reduction when message count is less than window size', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 10 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, mockAgent)

      expect(mockAgent.messages).toHaveLength(2)
    })

    it('skips reduction when message count equals window size', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, mockAgent)

      expect(mockAgent.messages).toHaveLength(2)
    })

    it('removes all messages when windowSize is 0', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 0 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, mockAgent)

      expect(mockAgent.messages).toHaveLength(0)
    })

    it('calls reduceContext when message count exceeds window size', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, mockAgent)

      // Should have trimmed; first message must be user
      expect(mockAgent.messages).toHaveLength(2)
      expect(mockAgent.messages[0]!.role).toBe('user')
    })
  })

  describe('reduceContext - tool result truncation', () => {
    it('partially truncates large tool results preserving first and last 200 chars', async () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const middle = 'MIDDLE_CONTENT_TO_REMOVE'.repeat(10) // 240 chars, safely above MIN_TRUNCATION_GAIN
      const original = 'A'.repeat(200) + middle + 'B'.repeat(200)
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock(original)],
            }),
          ],
        }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      const expectedText = `${'A'.repeat(200)}\n<truncated chars="${middle.length}"/>\n${'B'.repeat(200)}`
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock(expectedText)],
        })
      )
    })

    it('leaves small tool results unchanged', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock('Small result')],
            }),
          ],
        }),
      ]

      const result = (manager as any)._truncateToolResults(messages, 0)
      expect(result).toBe(false)
    })

    it('finds oldest message with tool results', async () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const firstOriginal = 'F'.repeat(500)
      const secondOriginal = 'S'.repeat(500)
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock(firstOriginal)],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-2',
              status: 'success',
              content: [new TextBlock(secondOriginal)],
            }),
          ],
        }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Oldest tool-result message is truncated; newer one is untouched.
      const expectedTruncated = `${'F'.repeat(200)}\n<truncated chars="100"/>\n${'F'.repeat(200)}`
      expect(messages[1]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock(expectedTruncated)],
        })
      )
      expect(messages[3]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-2',
          status: 'success',
          content: [new TextBlock(secondOriginal)],
        })
      )
    })

    it('returns after successful truncation without trimming messages', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: true })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock('L'.repeat(500))],
            }),
          ],
        }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should not have removed any messages, only truncated tool result
      expect(mockAgent.messages).toHaveLength(3)
    })

    it('skips truncation when shouldTruncateResults is false', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 3, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'tool-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock('L'.repeat(500))],
            }),
          ],
        }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should have trimmed messages instead of truncating tool result
      expect(mockAgent.messages).toHaveLength(3)
      expect(mockAgent.messages[0]!.role).toBe('user')

      // Tool result should not be truncated
      expect(mockAgent.messages[2]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('L'.repeat(500))],
        })
      )
    })

    it('does not re-truncate already-truncated results', async () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      // Produced by an earlier run: 200 chars + marker + 200 chars = well under the 450-char
      // threshold below which truncation is not worth running.
      const alreadyTruncated = 'A'.repeat(200) + '\n<truncated chars="1000"/>\n' + 'B'.repeat(200)
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock(alreadyTruncated)],
            }),
          ],
        }),
      ]

      // First call should return false (too short to gain anything from re-truncating)
      const result = (manager as any)._truncateToolResults(messages, 0)
      expect(result).toBe(false)

      // reduceContext should fall through to message trimming
      const messages2 = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new TextBlock(alreadyTruncated)],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
        new Message({ role: 'user', content: [new TextBlock('Message')] }),
      ]
      const mockAgent = createMockAgent({ messages: messages2 })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should have trimmed messages since truncation was skipped
      expect(mockAgent.messages.length).toBeLessThan(3)
    })

    it('replaces image blocks nested in tool results with descriptive placeholders', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const bytes = new Uint8Array(1234)
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new ImageBlock({ format: 'png', source: { bytes } }), new TextBlock('tail')],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)

      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[image: png, source: bytes, 1234 bytes]'), new TextBlock('tail')],
        })
      )
    })

    it('preserves the error field on truncated tool results', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const originalError = new Error('tool blew up')
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'error',
              content: [new TextBlock('x'.repeat(500))],
              error: originalError,
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)

      const expectedText = `${'x'.repeat(200)}\n<truncated chars="100"/>\n${'x'.repeat(200)}`
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'error',
          content: [new TextBlock(expectedText)],
          error: originalError,
        })
      )
    })

    it('image placeholder reflects non-bytes source kinds honestly', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [
                new ImageBlock({ format: 'jpeg', source: { url: 'https://example.com/x.jpg' } }),
                new ImageBlock({ format: 'png', source: { location: { type: 's3', uri: 's3://bucket/key' } } }),
              ],
            }),
          ],
        }),
      ]

      ;(manager as any)._truncateToolResults(messages, 0)

      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[image: jpeg, source: url]'), new TextBlock('[image: png, source: s3]')],
        })
      )
    })

    it('replaces video bytes blocks with a descriptive placeholder', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new VideoBlock({ format: 'mp4', source: { bytes: new Uint8Array(4096) } })],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[video: mp4, source: bytes, 4096 bytes]')],
        })
      )
    })

    it('replaces video s3 blocks with a descriptive placeholder', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [
                new VideoBlock({
                  format: 'mp4',
                  source: { location: { type: 's3', uri: 's3://bucket/key' } },
                }),
              ],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[video: mp4, source: s3]')],
        })
      )
    })

    it('replaces document bytes blocks with a descriptive placeholder', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [
                new DocumentBlock({
                  name: 'report',
                  format: 'pdf',
                  source: { bytes: new Uint8Array(8192) },
                }),
              ],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[document: report, pdf, source: bytes, 8192 bytes]')],
        })
      )
    })

    it('replaces document s3 blocks with a descriptive placeholder', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [
                new DocumentBlock({
                  name: 'spec',
                  format: 'pdf',
                  source: { location: { type: 's3', uri: 's3://b/k' } },
                }),
              ],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock('[document: spec, pdf, source: s3]')],
        })
      )
    })

    it('partially truncates large text inside a document text source', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const middle = 'M'.repeat(240)
      const originalText = 'A'.repeat(200) + middle + 'B'.repeat(200)
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new DocumentBlock({ name: 'report', format: 'txt', source: { text: originalText } })],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)

      const expectedText = `${'A'.repeat(200)}\n<truncated chars="${middle.length}"/>\n${'B'.repeat(200)}`
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new DocumentBlock({ name: 'report', format: 'txt', source: { text: expectedText } })],
        })
      )
    })

    it('leaves small text inside a document text source unchanged', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new DocumentBlock({ name: 'short', format: 'txt', source: { text: 'hello' } })],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(false)
    })

    it('truncates long nested text blocks inside a document content source', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const longText = 'A'.repeat(200) + 'M'.repeat(240) + 'B'.repeat(200)
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [
                new DocumentBlock({
                  name: 'pages',
                  format: 'txt',
                  source: { content: [new TextBlock(longText), new TextBlock('short')] },
                }),
              ],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)

      const expectedText = `${'A'.repeat(200)}\n<truncated chars="240"/>\n${'B'.repeat(200)}`
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [
            new DocumentBlock({
              name: 'pages',
              format: 'txt',
              source: { content: [new TextBlock(expectedText), new TextBlock('short')] },
            }),
          ],
        })
      )
    })

    it('replaces large json blocks with a size placeholder', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const big = { payload: 'x'.repeat(1000) }
      const size = JSON.stringify(big).length
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new JsonBlock({ json: big })],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(true)
      expect(messages[0]!.content[0]).toEqual(
        new ToolResultBlock({
          toolUseId: 'tool-1',
          status: 'success',
          content: [new TextBlock(`[json: ${size} chars]`)],
        })
      )
    })

    it('leaves small json blocks unchanged', () => {
      const manager = new SlidingWindowConversationManager({ shouldTruncateResults: true })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-1',
              status: 'success',
              content: [new JsonBlock({ json: { ok: true } })],
            }),
          ],
        }),
      ]

      const changed = (manager as any)._truncateToolResults(messages, 0)
      expect(changed).toBe(false)
    })

    it('does not call truncateToolResults unless an error is passed in', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: true })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Tool result content')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      // Spy on _truncateToolResults to verify it's NOT called
      const truncateSpy = vi.spyOn(manager as any, '_truncateToolResults')

      // Trigger window size enforcement (no error parameter)
      await triggerSlidingWindow(manager, mockAgent)

      // Verify _truncateToolResults was NOT called during window enforcement
      expect(truncateSpy).not.toHaveBeenCalled()

      // Should have trimmed; first message must be user
      expect(mockAgent.messages.length).toBe(1)
      expect(mockAgent.messages[0]!.role).toBe('user')

      truncateSpy.mockRestore()
    })
  })

  describe('reduceContext - message trimming', () => {
    it('trims oldest messages when tool results cannot be truncated', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 3, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 3')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      expect(mockAgent.messages).toHaveLength(3)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message 2' })
    })

    it('calculates correct trim index (messages.length - windowSize)', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should remove 2 messages (4 - 2 = 2)
      expect(mockAgent.messages).toHaveLength(2)
    })

    it('removes all messages when windowSize is 0 on context overflow', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 0, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      expect(mockAgent.messages).toHaveLength(0)
    })

    it('uses default trim index of 2 when messages <= windowSize', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 5 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should remove 2 messages (default when count <= windowSize)
      expect(mockAgent.messages).toHaveLength(1)
    })

    it('removes messages from start of array using splice', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Should keep last 2 messages
      expect(mockAgent.messages).toHaveLength(2)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message 2' })
      expect(mockAgent.messages[1]!.content[0]!).toEqual({ type: 'textBlock', text: 'Response 2' })
    })
  })

  describe('reduceContext - tool pair validation', () => {
    it('does not trim at index where oldest message is toolResult', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const messages = [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Result')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
        new Message({ role: 'user', content: [new TextBlock('Message')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Skips index 1 (toolResult) and index 2 (assistant), trims at index 3 (user)
      expect(mockAgent.messages).toHaveLength(1)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message' })
    })

    it('does not trim at index where oldest message is toolUse without following toolResult', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response')] }), // Not a toolResult
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // Skips index 1 (toolUse without following toolResult), skips index 2 (assistant), trims at index 3 (user)
      expect(mockAgent.messages).toHaveLength(1)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message 2' })
    })

    it('allows trim when oldest message is toolUse with following toolResult', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Result')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // trimIndex starts at 3 (5 - 2 = 3), which is assistant 'Response' — skipped (not user).
      // trimIndex 4 is user 'Message 2' — valid.
      expect(mockAgent.messages).toHaveLength(1)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message 2' })
    })

    it('allows trim at toolUse when toolResult immediately follows', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 4, shouldTruncateResults: false })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ name: 'tool1', toolUseId: 'id-1', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Result')],
            }),
          ],
        }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // trimIndex starts at 2 (6 - 4 = 2), which is user 'Message 2' — valid trim point
      expect(mockAgent.messages).toHaveLength(4)
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'Message 2' })
    })

    it('allows trim when oldest message is text or other non-tool content', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 2 })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
      ]
      const mockAgent = createMockAgent({ messages })

      await triggerContextOverflow(manager, mockAgent, new ContextWindowOverflowError('Context overflow'))

      // trimIndex starts at 2 (4 - 2 = 2), which is user 'Message 2' — valid
      expect(mockAgent.messages).toHaveLength(2)
      expect(mockAgent.messages[0]!.content[0]).toEqual({ type: 'textBlock', text: 'Message 2' })
    })

    it('skips assistant message to ensure trimmed conversation starts with user', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 8 })
      const messages = Array.from(
        { length: 9 },
        (_, i) => new Message({ role: i % 2 === 0 ? 'user' : 'assistant', content: [new TextBlock(`message ${i}`)] })
      )
      const mockAgent = createMockAgent({ messages })

      await triggerSlidingWindow(manager, mockAgent)

      // Naive trim would leave assistant at index 1 as first message.
      // Fix skips it so conversation starts with user at index 2.
      expect(mockAgent.messages[0]!.role).toBe('user')
      expect(mockAgent.messages[0]!.content[0]!).toEqual({ type: 'textBlock', text: 'message 2' })
    })

    it('returns false when no valid trim point exists', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 1, shouldTruncateResults: false })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Result')],
            }),
          ],
        }),
      ]

      const result = manager.reduce({
        agent: createMockAgent({ messages }),
        model: {} as Model,
        error: new ContextWindowOverflowError('Context overflow'),
      })

      expect(result).toBe(false)
    })

    it('propagates the original ContextWindowOverflowError when reduce cannot reduce further', async () => {
      const manager = new SlidingWindowConversationManager({ windowSize: 1, shouldTruncateResults: false })
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'id-1',
              status: 'success',
              content: [new TextBlock('Result')],
            }),
          ],
        }),
      ]
      const mockAgent = createMockAgent({ messages })
      const originalError = new ContextWindowOverflowError('Context overflow')

      // The base class hook does not set event.retry when reduce returns false,
      // so the original error propagates out of the hook chain
      const event = new AfterModelCallEvent({
        agent: mockAgent,
        model: {} as any,
        attemptCount: 1,
        error: originalError,
        invocationState: {},
      })
      const pluginAgent = createMockAgent()
      manager.initAgent(pluginAgent)
      await invokeTrackedHook(pluginAgent, event)

      expect(event.retry).toBeUndefined()
    })
  })

  describe('helper methods', () => {
    describe('findOldestMessageWithToolResults', () => {
      it('returns correct index when tool results exist', () => {
        const manager = new SlidingWindowConversationManager()
        const messages = [
          new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'id-1',
                status: 'success',
                content: [new TextBlock('Result 1')],
              }),
            ],
          }),
          new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
        ]

        const index = (manager as any)._findOldestMessageWithToolResults(messages)
        expect(index).toBe(1)
      })

      it('returns undefined when no tool results exist', () => {
        const manager = new SlidingWindowConversationManager()
        const messages = [
          new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
          new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        ]

        const index = (manager as any)._findOldestMessageWithToolResults(messages)
        expect(index).toBeUndefined()
      })

      it('iterates forward from start', () => {
        const manager = new SlidingWindowConversationManager()
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'id-1',
                status: 'success',
                content: [new TextBlock('Result 1')],
              }),
            ],
          }),
          new Message({ role: 'assistant', content: [new TextBlock('Response')] }),
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'id-2',
                status: 'success',
                content: [new TextBlock('Result 2')],
              }),
            ],
          }),
        ]

        const index = (manager as any)._findOldestMessageWithToolResults(messages)
        // Should find the first one (index 0), not the last (index 2)
        expect(index).toBe(0)
      })
    })

    describe('truncateToolResults', () => {
      it('returns true when changes are made', () => {
        const manager = new SlidingWindowConversationManager()
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'id-1',
                status: 'success',
                content: [new TextBlock('x'.repeat(500))],
              }),
            ],
          }),
        ]

        const result = (manager as any)._truncateToolResults(messages, 0)
        expect(result).toBe(true)
      })

      it('returns false when already truncated', () => {
        const manager = new SlidingWindowConversationManager()
        const alreadyTruncated = 'A'.repeat(200) + '\n<truncated chars="1000"/>\n' + 'B'.repeat(200)
        const messages = [
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'id-1',
                status: 'success',
                content: [new TextBlock(alreadyTruncated)],
              }),
            ],
          }),
        ]

        const result = (manager as any)._truncateToolResults(messages, 0)
        expect(result).toBe(false)
      })

      it('returns false when no tool results found', () => {
        const manager = new SlidingWindowConversationManager()
        const messages = [new Message({ role: 'user', content: [new TextBlock('Message')] })]

        const result = (manager as any)._truncateToolResults(messages, 0)
        expect(result).toBe(false)
      })
    })
  })

  describe('reduceOnThreshold', () => {
    it('trims oldest messages when compressionThreshold is exceeded', async () => {
      const manager = new SlidingWindowConversationManager({
        windowSize: 4,
        proactiveCompression: { compressionThreshold: 0.7 },
      })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 2')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 2')] }),
        new Message({ role: 'user', content: [new TextBlock('Message 3')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 3')] }),
      ]
      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any
      const mockAgent = createMockAgent({ messages })
      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 800,
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages.length).toBe(4)
    })

    it('does not trim when below compressionThreshold', async () => {
      const manager = new SlidingWindowConversationManager({
        windowSize: 4,
        proactiveCompression: { compressionThreshold: 0.7 },
      })
      const messages = [
        new Message({ role: 'user', content: [new TextBlock('Message 1')] }),
        new Message({ role: 'assistant', content: [new TextBlock('Response 1')] }),
      ]
      const mockModel = { getConfig: () => ({ contextWindowLimit: 1000 }) as BaseModelConfig } as any
      const mockAgent = createMockAgent({ messages })
      manager.initAgent(mockAgent)

      const event = new BeforeModelCallEvent({
        agent: mockAgent,
        model: mockModel,
        invocationState: {},
        projectedInputTokens: 500,
      })
      await invokeTrackedHook(mockAgent, event)

      expect(mockAgent.messages).toHaveLength(2)
    })
  })
})
