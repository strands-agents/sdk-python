import { describe, it, expect, vi } from 'vitest'
import { Message, TextBlock, ToolResultBlock } from '../../types/messages.js'
import { foldIntoLastUserMessage, isUserTurn, resolveTrigger, createInjectionMiddleware } from '../message-injection.js'
import type { InvokeModelContext } from '../../middleware/index.js'
import { logger } from '../../logging/logger.js'

const user = (text: string) => new Message({ role: 'user', content: [new TextBlock(text)] })
const assistant = (text: string) => new Message({ role: 'assistant', content: [new TextBlock(text)] })
const toolResult = () =>
  new Message({
    role: 'user',
    content: [new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('done')] })],
  })
describe('foldIntoLastUserMessage', () => {
  it('prepends the text as a leading TextBlock on the last user message, ahead of its content', () => {
    const messages = [user('original task'), assistant('prior step'), user('next ask')]
    const result = foldIntoLastUserMessage(messages, 'INJECTED')

    expect(result).toHaveLength(3)
    expect(result.map((m) => m.role)).toStrictEqual(['user', 'assistant', 'user'])
    const target = result[2]!
    expect(target.content).toHaveLength(2)
    expect(target.content[0]).toBeInstanceOf(TextBlock)
    expect((target.content[0] as TextBlock).text).toBe('INJECTED')
    expect((target.content[1] as TextBlock).text).toBe('next ask')
    // The user message stays last so the user's ask remains in the recency slot.
    expect(result[result.length - 1]).toBe(target)
  })

  it('returns a new array and does not mutate the input or its messages', () => {
    const original = user('ask')
    const messages = [assistant('prior'), original]
    const result = foldIntoLastUserMessage(messages, 'INJECTED')

    expect(result).not.toBe(messages)
    expect(messages[1]).toBe(original)
    expect(original.content).toHaveLength(1) // untouched
    expect(result[1]).not.toBe(original)
  })

  it('folds into a tool-result user message ahead of the tool result block', () => {
    const messages = [user('task'), assistant('thinking'), toolResult()]
    const result = foldIntoLastUserMessage(messages, 'INJECTED')

    const target = result[2]!
    expect(target.content[0]).toBeInstanceOf(TextBlock)
    expect((target.content[0] as TextBlock).text).toBe('INJECTED')
    expect(target.content[1]!.type).toBe('toolResultBlock')
  })

  it('targets the most recent user message when several exist', () => {
    const messages = [user('first'), assistant('a'), user('second')]
    const result = foldIntoLastUserMessage(messages, 'INJECTED')

    expect((result[0]!.content[0] as TextBlock).text).toBe('first') // earlier user untouched
    expect((result[2]!.content[0] as TextBlock).text).toBe('INJECTED')
  })

  it('preserves message metadata on the folded message', () => {
    const tagged = new Message({
      role: 'user',
      content: [new TextBlock('ask')],
      metadata: { custom: { keep: 'me' } },
    })
    const result = foldIntoLastUserMessage([tagged], 'INJECTED')
    expect(result[0]!.metadata?.custom).toStrictEqual({ keep: 'me' })
  })

  it('returns the input unchanged when there is no user message', () => {
    const messages = [assistant('only assistant')]
    const result = foldIntoLastUserMessage(messages, 'INJECTED')
    expect(result).toBe(messages)
  })
})

describe('isUserTurn', () => {
  it('is true when the last message is a plain user ask', () => {
    expect(isUserTurn([assistant('prior').toJSON(), user('ask').toJSON()])).toBe(true)
  })

  it('is false when the last message is a user tool-result turn', () => {
    expect(isUserTurn([user('task').toJSON(), assistant('a').toJSON(), toolResult().toJSON()])).toBe(false)
  })

  it('is false when the last message is an assistant message', () => {
    expect(isUserTurn([user('ask').toJSON(), assistant('reply').toJSON()])).toBe(false)
  })

  it('is false for an empty conversation', () => {
    expect(isUserTurn([])).toBe(false)
  })
})

describe('resolveTrigger', () => {
  it('defaults (undefined) to the userTurn policy', () => {
    const trigger = resolveTrigger(undefined)
    expect(trigger([user('ask').toJSON()])).toBe(true)
    expect(trigger([toolResult().toJSON()])).toBe(false)
  })

  it("'userTurn' uses isUserTurn", () => {
    const trigger = resolveTrigger('userTurn')
    expect(trigger([user('ask').toJSON()])).toBe(true)
    expect(trigger([toolResult().toJSON()])).toBe(false)
  })

  it("'everyTurn' always fires", () => {
    const trigger = resolveTrigger('everyTurn')
    expect(trigger([])).toBe(true)
    expect(trigger([toolResult().toJSON()])).toBe(true)
  })

  it('uses a custom predicate', () => {
    const trigger = resolveTrigger((messages) => messages.length >= 2)
    expect(trigger([user('a').toJSON()])).toBe(false)
    expect(trigger([user('a').toJSON(), assistant('b').toJSON()])).toBe(true)
  })

  it('fails open (returns false, logs) when a custom predicate throws', () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const trigger = resolveTrigger(() => {
      throw new Error('boom')
    })
    expect(trigger([user('ask').toJSON()])).toBe(false)
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})

describe('createInjectionMiddleware', () => {
  // The handler is an InvokeModelStage.Input transformer: it only reads `context.messages` and spreads
  // the rest through, so a context carrying just `messages` exercises it faithfully.
  const ctx = (messages: Message[]) => ({ messages }) as unknown as InvokeModelContext

  it('folds provide() text into the latest user message, leaving other context fields intact', async () => {
    const handler = createInjectionMiddleware({ provide: async () => 'INJECTED' })
    const result = await handler(ctx([assistant('prior'), user('ask')]))

    expect(result.messages).toHaveLength(2)
    const target = result.messages[1]!
    expect((target.content[0] as TextBlock).text).toBe('INJECTED')
    expect((target.content[1] as TextBlock).text).toBe('ask')
  })

  it('passes the conversation (as data) to provide', async () => {
    const seen: string[] = []
    const handler = createInjectionMiddleware({
      provide: async (messages) => {
        seen.push(...messages.map((m) => m.role))
        return 'x'
      },
    })
    await handler(ctx([assistant('prior'), user('ask')]))

    expect(seen).toStrictEqual(['assistant', 'user'])
  })

  it('returns the context unchanged when the trigger does not fire', async () => {
    const provide = vi.fn(async () => 'x')
    const handler = createInjectionMiddleware({ provide }) // default 'userTurn'
    const input = ctx([user('task'), assistant('a'), toolResult()])
    const result = await handler(input)

    expect(result).toBe(input)
    expect(provide).not.toHaveBeenCalled()
  })

  it("'everyTurn' injects on an autonomous tool-result turn", async () => {
    const handler = createInjectionMiddleware({ trigger: 'everyTurn', provide: async () => 'INJECTED' })
    const result = await handler(ctx([user('task'), assistant('a'), toolResult()]))

    // The most recent user message on a tool-result turn is the tool-result itself; the fold prepends
    // ahead of its tool-result block.
    const folded = result.messages[2]!
    expect((folded.content[0] as TextBlock).text).toBe('INJECTED')
    expect(folded.content[1]!.type).toBe('toolResultBlock')
  })

  it('returns the context unchanged when provide yields empty text', async () => {
    const handler = createInjectionMiddleware({ provide: async () => '   ' })
    const input = ctx([assistant('prior'), user('ask')])
    const result = await handler(input)

    expect(result).toBe(input)
  })

  it('fails open (returns the context unchanged, logs) when provide throws', async () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const handler = createInjectionMiddleware({
      provide: async () => {
        throw new Error('boom')
      },
    })
    const input = ctx([assistant('prior'), user('ask')])
    const result = await handler(input)

    expect(result).toBe(input)
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })

  it('does not mutate the original context messages', async () => {
    const handler = createInjectionMiddleware({ provide: async () => 'INJECTED' })
    const input = ctx([assistant('prior'), user('ask')])
    const before = input.messages[1]!
    await handler(input)

    expect(before.content).toHaveLength(1) // the original user message is untouched
  })
})
