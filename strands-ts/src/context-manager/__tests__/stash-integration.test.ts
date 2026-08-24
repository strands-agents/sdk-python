import { describe, it, expect } from 'vitest'
import { Offload } from '../strategies/offload/index.js'
import { Stash } from '../stash.js'
import { RETRIEVAL_TOOL_NAME } from '../retrieval-tool.js'
import { InMemoryStorage } from '../../storage/in-memory-storage.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { ImageBlock } from '../../types/media.js'
import { createMockAgent } from '../../__fixtures__/agent-helpers.js'
import type { ContextState } from '../types.js'

function makeToolResultMessage(text: string, toolUseId = 'tool-123'): Message {
  return new Message({
    role: 'user',
    content: [
      new ToolResultBlock({
        toolUseId,
        status: 'success',
        content: [new TextBlock(text)],
      }),
    ],
  })
}

function heuristicCountTokens(messages: Message[]): number {
  let total = 0
  for (const message of messages) {
    for (const block of message.content) {
      if (block instanceof TextBlock) {
        total += Math.ceil(block.text.length / 4)
      } else if (block instanceof ToolResultBlock) {
        for (const content of block.content) {
          if (content instanceof TextBlock) total += Math.ceil(content.text.length / 4)
          else total += Math.ceil(JSON.stringify(content).length / 2)
        }
      } else {
        total += Math.ceil(JSON.stringify(block).length / 2)
      }
    }
  }
  return total
}

async function stashAll(stash: Stash, messages: Message[]): Promise<void> {
  for (const message of messages) await stash.storeMessage(message)
}

function makeContext(messages: Message[], stash?: Stash, utilization = 0.5): ContextState {
  const agent = createMockAgent({
    messages,
    extra: { model: { countTokens: async (msgs: Message[]) => heuristicCountTokens(msgs) } } as never,
  })
  const base: ContextState = {
    messages,
    agent,
    utilization,
  }
  if (stash) {
    return { ...base, stash }
  }
  return base
}

describe('Offload strategies with stash', () => {
  describe('truncate + stash', () => {
    it('persists original content to stash before truncating', async () => {
      const storage = new InMemoryStorage()
      const stash = new Stash(storage)
      const largeText = 'important data '.repeat(1000)
      const messages = [makeToolResultMessage(largeText)]
      await stashAll(stash, messages)
      const strategy = Offload.truncate('toolResults')
      const context = makeContext(messages, stash)

      const acted = await strategy.apply(context)

      expect(acted).toBe(true)

      const keys = await stash.list()
      expect(keys.length).toBe(1)

      const retrieved = await stash.retrieve(keys[0]!)
      expect(retrieved).not.toBeNull()
      expect(new TextDecoder().decode(retrieved!.content)).toBe(largeText)
    })

    it('includes stash reference in the truncated preview', async () => {
      const stash = new Stash(new InMemoryStorage())
      const largeText = 'x'.repeat(20000)
      const messages = [makeToolResultMessage(largeText)]
      await stashAll(stash, messages)
      const strategy = Offload.truncate('toolResults')
      const context = makeContext(messages, stash)

      await strategy.apply(context)

      const block = messages[0]!.content[0] as ToolResultBlock
      const text = (block.content[0] as TextBlock).text
      expect(text).toContain('[Stashed: ref:')
    })

    it('does not stash when no stash is configured', async () => {
      const largeText = 'x'.repeat(20000)
      const messages = [makeToolResultMessage(largeText)]
      const strategy = Offload.truncate('toolResults')
      const context = makeContext(messages)

      await strategy.apply(context)

      const block = messages[0]!.content[0] as ToolResultBlock
      const text = (block.content[0] as TextBlock).text
      expect(text).not.toContain('[Stashed:')
    })
  })

  describe('drop + stash', () => {
    it('persists original content before dropping', async () => {
      const stash = new Stash(new InMemoryStorage())
      const largeText = 'critical information '.repeat(500)
      const messages = [makeToolResultMessage(largeText)]
      await stashAll(stash, messages)
      const strategy = Offload.drop('toolResults')
      const context = makeContext(messages, stash)

      const acted = await strategy.apply(context)

      expect(acted).toBe(true)

      const keys = await stash.list()
      expect(keys.length).toBe(1)

      const retrieved = await stash.retrieve(keys[0]!)
      expect(new TextDecoder().decode(retrieved!.content)).toBe(largeText)
    })

    it('includes stash reference in the drop marker', async () => {
      const stash = new Stash(new InMemoryStorage())
      const largeText = 'x'.repeat(20000)
      const messages = [makeToolResultMessage(largeText)]
      await stashAll(stash, messages)
      const strategy = Offload.drop('toolResults')
      const context = makeContext(messages, stash)

      await strategy.apply(context)

      const block = messages[0]!.content[0] as ToolResultBlock
      const text = (block.content[0] as TextBlock).text
      expect(text).toContain('[Dropped] ref:')
    })
  })

  describe('binary content stashing', () => {
    it('stashes image content from tool results', async () => {
      const stash = new Stash(new InMemoryStorage())
      const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-img',
              status: 'success',
              content: [new ImageBlock({ format: 'png', source: { bytes: imageBytes } })],
            }),
          ],
        }),
      ]
      await stashAll(stash, messages)
      const strategy = Offload.drop('toolResults')
      const context = makeContext(messages, stash)

      const acted = await strategy.apply(context)

      expect(acted).toBe(true)
      const keys = await stash.list()
      expect(keys.length).toBe(1)

      const retrieved = await stash.retrieve(keys[0]!)
      expect(retrieved!.contentType).toBe('image/png')
      expect(retrieved!.content).toEqual(imageBytes)
    })

    it('stashes each block independently when mixed content', async () => {
      const stash = new Stash(new InMemoryStorage())
      const imageBytes = new Uint8Array([0xff, 0xd8])
      const messages = [
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'tool-mixed',
              status: 'success',
              content: [
                new TextBlock('description of image'),
                new ImageBlock({ format: 'jpeg', source: { bytes: imageBytes } }),
              ],
            }),
          ],
        }),
      ]
      await stashAll(stash, messages)
      const strategy = Offload.drop('toolResults')
      const context = makeContext(messages, stash)

      await strategy.apply(context)

      const keys = await stash.list()
      expect(keys.length).toBe(2)

      const entries = await Promise.all(keys.map((key) => stash.retrieve(key)))
      const textEntry = entries.find((entry) => entry!.contentType === 'text/plain')
      const imageEntry = entries.find((entry) => entry!.contentType === 'image/jpeg')

      expect(textEntry).not.toBeNull()
      expect(new TextDecoder().decode(textEntry!.content)).toBe('description of image')

      expect(imageEntry).not.toBeNull()
      expect(imageEntry!.content).toEqual(imageBytes)
    })
  })

  describe('eager stashing via storeMessage', () => {
    it('persists tool result content on message arrival', async () => {
      const stash = new Stash(new InMemoryStorage())
      const largeText = 'important data '.repeat(500)
      const message = makeToolResultMessage(largeText, 'call-1')

      await stash.storeMessage(message)

      const keys = await stash.list()
      expect(keys.length).toBe(1)
      const retrieved = await stash.retrieve(keys[0]!)
      expect(retrieved).not.toBeNull()
      expect(new TextDecoder().decode(retrieved!.content)).toBe(largeText)
    })

    it('persists text blocks from assistant messages on arrival', async () => {
      const stash = new Stash(new InMemoryStorage())
      const assistantText = 'important analysis '.repeat(200)
      const message = new Message({ role: 'assistant', content: [new TextBlock(assistantText)] })

      await stash.storeMessage(message)

      const keys = await stash.list()
      expect(keys.length).toBe(1)
      const retrieved = await stash.retrieve(keys[0]!)
      expect(new TextDecoder().decode(retrieved!.content)).toBe(assistantText)
    })
  })

  describe('retrieval loop prevention', () => {
    it('does not offload retrieve_context tool results when stash is active', async () => {
      const stash = new Stash(new InMemoryStorage())
      const largeRetrievalResult = 'retrieved content '.repeat(1000)

      const messages = [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'call-1', name: RETRIEVAL_TOOL_NAME, input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call-1',
              status: 'success',
              content: [new TextBlock(largeRetrievalResult)],
            }),
          ],
        }),
      ]
      const strategy = Offload.truncate('toolResults')
      const context = makeContext(messages, stash)

      const acted = await strategy.apply(context)

      expect(acted).toBe(false)
      const block = messages[1]!.content[0] as ToolResultBlock
      const text = (block.content[0] as TextBlock).text
      expect(text).toBe(largeRetrievalResult)
    })

    it('offloads retrieve_context results when no stash (stash not configured)', async () => {
      const largeRetrievalResult = 'retrieved content '.repeat(1000)

      const messages = [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'call-1', name: RETRIEVAL_TOOL_NAME, input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call-1',
              status: 'success',
              content: [new TextBlock(largeRetrievalResult)],
            }),
          ],
        }),
      ]
      const strategy = Offload.truncate('toolResults')
      const context = makeContext(messages)

      const acted = await strategy.apply(context)

      expect(acted).toBe(true)
    })
  })
})
