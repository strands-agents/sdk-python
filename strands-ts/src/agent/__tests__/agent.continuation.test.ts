import { describe, expect, it, vi } from 'vitest'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { AfterInvocationEvent, BeforeModelCallEvent, MessageAddedEvent } from '../../hooks/events.js'
import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { Agent } from '../agent.js'
import { continuations } from '../continuation.js'

function textOf(message: Message): string {
  return message.content.flatMap((block) => (block.type === 'textBlock' ? [block.text] : [])).join('')
}

function textModel(...turns: string[]): MockMessageModel {
  return turns.reduce((model, text) => model.addTurn({ type: 'textBlock', text }), new MockMessageModel())
}

function captureRequests(model: MockMessageModel): Message[][] {
  const requests: Message[][] = []
  const originalStream = model.stream.bind(model)
  vi.spyOn(model, 'stream').mockImplementation(async function* (messages, options) {
    requests.push(messages.map((message) => message.clone()))
    yield* originalStream(messages, options)
  })
  return requests
}

describe('Agent continuation input', () => {
  it('combines multiple follow-up contributions with public resume input', async () => {
    const model = textModel('initial', 'final')
    const requests = captureRequests(model)
    const appended: string[] = []
    const abandoned = vi.fn()
    const agent = new Agent({ model, printer: false })
    let resumed = false

    agent.addHook(AfterInvocationEvent, (event) => {
      if (resumed) return
      resumed = true
      for (const args of ['first', 'second']) {
        continuations.addInput(event, {
          args,
          onAppended: () => {
            appended.push(args)
          },
        })
      }
      continuations.addInput(event, {
        args: [new Message({ role: 'assistant', content: [new TextBlock('invalid')] })],
        onAbandoned: abandoned,
      })
      event.resume = 'public'
    })

    await agent.invoke('start')

    expect(requests[1]?.map((message) => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(textOf(requests[1]!.at(-1)!)).toBe('firstsecondpublic')
    expect(agent.messages.map(textOf)).toEqual(['start', 'initial', 'firstsecondpublic', 'final'])
    expect(appended).toEqual(['first', 'second'])
    expect(abandoned).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Continuation input must contain a complete message sequence' })
    )
  })

  it('appends a complete tool exchange contributed before the model call', async () => {
    const model = textModel('final')
    const requests = captureRequests(model)
    const metadata = { custom: { pinned: true } }
    const initialInput = new Message({
      role: 'user',
      content: [new TextBlock('start')],
      trackingId: 'durable-1',
      metadata,
    })
    const toolUse = new Message({
      role: 'assistant',
      content: [new ToolUseBlock({ name: 'background-result', toolUseId: 'delivery-1', input: {} })],
    })
    const toolResult = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'delivery-1',
          status: 'success',
          content: [new TextBlock('complete')],
        }),
      ],
    })
    const appended = vi.fn()
    const addedMessages: Message[] = []
    const agent = new Agent({ model, printer: false })

    agent.addHook(MessageAddedEvent, (event) => {
      addedMessages.push(event.message)
    })
    agent.addHook(BeforeModelCallEvent, (event) => {
      continuations.addInput(event, { args: 'guidance' })
      continuations.addInput(event, {
        args: [toolUse, toolResult],
        onAppended: () => {
          expect(agent.messages).toEqual([expect.any(Message), toolUse, toolResult])
          appended()
        },
      })
    })

    await agent.invoke([initialInput])

    expect(requests[0]?.map((message) => message.role)).toEqual(['user', 'assistant', 'user'])
    expect(textOf(requests[0]![0]!)).toBe('startguidance')
    expect(agent.messages).toEqual([
      new Message({
        role: 'user',
        content: [new TextBlock('start'), new TextBlock('guidance')],
        trackingId: initialInput.trackingId,
        metadata,
      }),
      toolUse,
      toolResult,
      expect.any(Message),
    ])
    expect(addedMessages).toContain(agent.messages[0])
    expect(appended).toHaveBeenCalledOnce()
  })

  it('abandons follow-up input when the stream closes before it is appended', async () => {
    const abandoned = vi.fn()
    const agent = new Agent({ model: textModel('initial', 'unreachable'), printer: false })
    let added = false

    agent.addHook(AfterInvocationEvent, (event) => {
      if (added) return
      added = true
      continuations.addInput(event, { args: 'pending', onAbandoned: abandoned })
    })

    for await (const event of agent.stream('start')) {
      if (event instanceof MessageAddedEvent) break
    }

    expect(abandoned).toHaveBeenCalledOnce()
    expect(agent.messages.map(textOf)).toEqual(['start'])
  })
})
